import pylangacq
import os
import re
import subprocess
import shutil

def check_ffmpeg_availability():
    if shutil.which("ffmpeg") is None:
        print("--------------------------------------------------------------------")
        print("Error: ffmpeg command not found.")
        print("Please install ffmpeg and ensure it is in your system's PATH.")
        print("You can download ffmpeg from: https://ffmpeg.org/download.html")
        print("--------------------------------------------------------------------")
        return False
    return True

def final_cleanup_joined_transcript(text):
    # Squeeze multiple spaces that might result from joining words,
    # especially if some words were entirely removed (e.g., were only punctuation).
    processed_text = re.sub(r"\s+", " ", text).strip()
    return processed_text

def process_talkbank_files():
    if not check_ffmpeg_availability():
        return

    cha_dir = "."
    mp4_dir = "."
    output_dir = "output_par_combined"
    os.makedirs(output_dir, exist_ok=True)

    temp_audio_main_dir = os.path.join(output_dir, "_temp_audio_segments")
    os.makedirs(temp_audio_main_dir, exist_ok=True)

    all_cha_files = [f for f in os.listdir(cha_dir) if f.lower().endswith(".cha")]

    if not all_cha_files:
        print("No .cha files found in the current directory.")
        if os.path.exists(temp_audio_main_dir): shutil.rmtree(temp_audio_main_dir)
        return

    print(f"Found CHA files: {all_cha_files}")

    for cha_filename in all_cha_files:
        cha_file_basename_original = os.path.splitext(cha_filename)[0]
        
        mp4_target_basename = None
        match = re.search(r"aprocsa(\d{4})a", cha_file_basename_original, re.IGNORECASE)
        if match:
            mp4_target_basename = match.group(1)
        else:
            print(f"Warning: CHA filename '{cha_filename}' does not match 'aprocsaDDDDa' pattern. Skipping.")
            continue
        
        potential_mp4_files = [f for f in os.listdir(mp4_dir) 
                               if os.path.splitext(f)[0] == mp4_target_basename and f.lower().endswith(".mp4")]
        
        if not potential_mp4_files:
            print(f"Warning: MP4 file like '{mp4_target_basename}.mp4' (for '{cha_filename}') not found. Skipping.")
            continue
        
        mp4_filename_actual = potential_mp4_files[0]
        mp4_path = os.path.join(mp4_dir, mp4_filename_actual)
        cha_path = os.path.join(cha_dir, cha_filename)

        print(f"\nProcessing video file associated with: {cha_filename} (MP4: {mp4_path})")

        try:
            reader = pylangacq.read_chat(cha_path)
        except Exception as e:
            print(f"Error reading CHA file {cha_path}: {e}")
            continue

        all_par_transcripts_for_this_video = []
        temp_audio_segment_paths_for_this_video = []
        par_utterance_temp_file_idx = 0

        for utterance in reader.utterances():
            if utterance.participant == "PAR":
                if not utterance.time_marks:
                    continue

                start_ms, end_ms = utterance.time_marks
                if start_ms is None or end_ms is None or start_ms >= end_ms:
                    continue
                    
                start_sec = start_ms / 1000.0
                end_sec = end_ms / 1000.0

                cleaned_words_for_utterance = []
                for token in utterance.tokens:
                    word = token.word 

                    if not word: 
                        continue

                    # --- NEW CLEANING RULES APPLIED TO EACH WORD ---
                    # 1. Remove specific unwanted whole words (case-sensitive)
                    if word == "POSTCLITIC":
                        continue
                    
                    # 2. Remove specific symbols like "‡"
                    word = word.replace("‡", "")

                    # 3. Filter standard CHAT markers (these are usually standalone tokens)
                    if word.startswith(('&-', '&=')) or \
                       word in ["(.)", "(..)", "(...)"] or \
                       word.startswith(('[%', '[+', '[*', '<', '>')) or \
                       word.lower() == 'xxx': # Add more CHAT markers to filter if needed
                        continue
                    
                    # 4. Punctuation removal: Keep letters, numbers, and apostrophes only.
                    #    \w matches letters, numbers, and underscore. ' matches apostrophe.
                    #    This will remove periods, commas, question marks, etc.
                    word = re.sub(r"[^\w']", "", word)
                    # --- END OF NEW CLEANING RULES ---

                    if not word: # If the word became empty after cleaning (e.g., it was only "."), skip it
                        continue
                    
                    cleaned_words_for_utterance.append(word)
                
                transcript_for_this_utterance = " ".join(cleaned_words_for_utterance)
                # final_cleanup_joined_transcript now mostly handles extra spaces
                transcript_for_this_utterance = final_cleanup_joined_transcript(transcript_for_this_utterance)

                if not transcript_for_this_utterance:
                    continue
                
                all_par_transcripts_for_this_video.append(transcript_for_this_utterance)
                
                par_utterance_temp_file_idx += 1
                temp_segment_filename = f"{cha_file_basename_original}_par_temp_{par_utterance_temp_file_idx}.wav"
                temp_segment_path = os.path.join(temp_audio_main_dir, temp_segment_filename)

                ffmpeg_extract_cmd = [
                    'ffmpeg', '-i', mp4_path, '-ss', str(start_sec), '-to', str(end_sec),
                    '-vn', '-acodec', 'pcm_s16le', '-y', temp_segment_path
                ]
                try:
                    subprocess.run(ffmpeg_extract_cmd, check=True, capture_output=True, text=True)
                    temp_audio_segment_paths_for_this_video.append(temp_segment_path)
                except subprocess.CalledProcessError as e_extract:
                    print(f"  Error extracting audio segment for utterance {par_utterance_temp_file_idx} ({start_sec:.2f}s-{end_sec:.2f}s): {e_extract.stderr.strip()}")
                except Exception as e_gen_extract:
                     print(f"  Unexpected error extracting audio segment {par_utterance_temp_file_idx}: {e_gen_extract}")

        output_base_filename_for_video = f"{cha_file_basename_original}_PAR"
        if all_par_transcripts_for_this_video:
            combined_txt_path = os.path.join(output_dir, f"{output_base_filename_for_video}.txt")
            with open(combined_txt_path, "w", encoding="utf-8") as f_txt:
                f_txt.write("\n\n".join(all_par_transcripts_for_this_video))
            print(f"  Saved combined transcript: {combined_txt_path}")
        else:
            print(f"  No PAR transcripts found/extracted for {cha_filename}.")

        if temp_audio_segment_paths_for_this_video:
            combined_wav_path = os.path.join(output_dir, f"{output_base_filename_for_video}.wav")
            concat_list_filename = f"{cha_file_basename_original}_concat_list.txt"
            concat_list_path = os.path.join(temp_audio_main_dir, concat_list_filename)

            with open(concat_list_path, "w", encoding="utf-8") as f_clist:
                for temp_wav_path in temp_audio_segment_paths_for_this_video:
                    f_clist.write(f"file '{os.path.abspath(temp_wav_path)}'\n")
            
            ffmpeg_concat_cmd = [
                'ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_list_path,
                '-c', 'copy', '-y', combined_wav_path
            ]
            try:
                subprocess.run(ffmpeg_concat_cmd, check=True, capture_output=True, text=True)
                print(f"  Saved combined audio: {combined_wav_path}")
            except subprocess.CalledProcessError as e_concat:
                print(f"  Error concatenating audio for {cha_file_basename_original}: {e_concat.stderr.strip()}")
            except Exception as e_gen_concat:
                print(f"  Unexpected error concatenating audio for {cha_file_basename_original}: {e_gen_concat}")
            finally:
                if os.path.exists(concat_list_path):
                    try: os.remove(concat_list_path)
                    except OSError as e_rm_clist: print(f"  Warning: Could not remove temp concat list {concat_list_path}: {e_rm_clist}")
        else:
            if all_par_transcripts_for_this_video:
                 print(f"  No PAR audio segments successfully extracted for {cha_filename} to combine.")

        for temp_segment_file_path in temp_audio_segment_paths_for_this_video:
            if os.path.exists(temp_segment_file_path):
                try: os.remove(temp_segment_file_path)
                except OSError as e_rm_segment: print(f"  Warning: Could not remove temp audio segment {temp_segment_file_path}: {e_rm_segment}")
        
        print(f"  Finished processing for {cha_filename}.")

    try:
        if os.path.exists(temp_audio_main_dir) and not os.listdir(temp_audio_main_dir):
            shutil.rmtree(temp_audio_main_dir)
            print(f"\nCleaned up empty temporary directory: {temp_audio_main_dir}")
    except OSError as e_rm_main_temp:
        print(f"\nWarning: Could not remove main temporary directory {temp_audio_main_dir}: {e_rm_main_temp}")

    print(f"\nProcessing complete. Combined outputs are in '{output_dir}' directory.")

if __name__ == "__main__":
    process_talkbank_files()