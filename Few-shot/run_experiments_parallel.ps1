# Get and display the current directory
$currentPath = Get-Location
Write-Host "Current directory: $currentPath"

# First batch: All exhaustive_phoneme experiments
$commands_exhaustive = @(
    # AWS
    'python run_per_asr_experiments.py --single-asr "AWS" --single-strategy "exhaustive_phoneme" --runs 1',

    # AssemblyAI
    'python run_per_asr_experiments.py --single-asr "AssemblyAI" --single-strategy "exhaustive_phoneme" --runs 1',

    # Azure
    'python run_per_asr_experiments.py --single-asr "Azure" --single-strategy "exhaustive_phoneme" --runs 1',

    # Deepgram
    'python run_per_asr_experiments.py --single-asr "Deepgram" --single-strategy "exhaustive_phoneme" --runs 1',

    # ElevenLabs
    'python run_per_asr_experiments.py --single-asr "ElevenLabs" --single-strategy "exhaustive_phoneme" --runs 1',

    # GCP
    'python run_per_asr_experiments.py --single-asr "GCP" --single-strategy "exhaustive_phoneme" --runs 1',

    # Gemini
    'python run_per_asr_experiments.py --single-asr "Gemini" --single-strategy "exhaustive_phoneme" --runs 1',

    # Gladia
    'python run_per_asr_experiments.py --single-asr "Gladia" --single-strategy "exhaustive_phoneme" --runs 1',

    # Speechmatics
    'python run_per_asr_experiments.py --single-asr "Speechmatics" --single-strategy "exhaustive_phoneme" --runs 1',

    # Whisper
    'python run_per_asr_experiments.py --single-asr "Whisper" --single-strategy "exhaustive_phoneme" --runs 1'
)

# Second batch: All random_error experiments
$commands_random = @(
    # AWS
    'python run_per_asr_experiments.py --single-asr "AWS" --single-strategy "random_error" --runs 1',

    # AssemblyAI
    'python run_per_asr_experiments.py --single-asr "AssemblyAI" --single-strategy "random_error" --runs 1',

    # Azure
    'python run_per_asr_experiments.py --single-asr "Azure" --single-strategy "random_error" --runs 1',

    # Deepgram
    'python run_per_asr_experiments.py --single-asr "Deepgram" --single-strategy "random_error" --runs 1',

    # ElevenLabs
    'python run_per_asr_experiments.py --single-asr "ElevenLabs" --single-strategy "random_error" --runs 1',

    # GCP
    'python run_per_asr_experiments.py --single-asr "GCP" --single-strategy "random_error" --runs 1',

    # Gemini
    'python run_per_asr_experiments.py --single-asr "Gemini" --single-strategy "random_error" --runs 1',

    # Gladia
    'python run_per_asr_experiments.py --single-asr "Gladia" --single-strategy "random_error" --runs 1',

    # Speechmatics
    'python run_per_asr_experiments.py --single-asr "Speechmatics" --single-strategy "random_error" --runs 1',

    # Whisper
    'python run_per_asr_experiments.py --single-asr "Whisper" --single-strategy "random_error" --runs 1'
)

# Verify the Python script exists
if (-not (Test-Path "run_per_asr_experiments.py")) {
    Write-Host "Error: run_per_asr_experiments.py not found in current directory!"
    Write-Host "Please make sure you're running this script from the correct directory."
    exit 1
}

# Run commands in batches of 10
$batchSize = 10

Write-Host "`n=== Starting exhaustive_phoneme experiments ==="
$totalCommands = $commands_exhaustive.Count
$currentBatch = 0

while ($currentBatch -lt $totalCommands) {
    Write-Host "`nStarting batch $([Math]::Floor($currentBatch / $batchSize) + 1) of $([Math]::Ceiling($totalCommands / $batchSize))"
    
    # Start the next batch of processes
    for ($i = 0; $i -lt $batchSize -and ($currentBatch + $i) -lt $totalCommands; $i++) {
        $command = $commands_exhaustive[$currentBatch + $i]
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$currentPath'; Write-Host 'Running: $command'; echo y | $command"
        Start-Sleep -Seconds 2
    }
    
    # Move to next batch
    $currentBatch += $batchSize
}

Write-Host "`nAll exhaustive_phoneme experiments started. Press Enter when they're done to start random_error experiments, or 'q' to quit..."
$response = Read-Host
if ($response -ne 'q') {
    Write-Host "`n=== Starting random_error experiments ==="
    $totalCommands = $commands_random.Count
    $currentBatch = 0

    while ($currentBatch -lt $totalCommands) {
        Write-Host "`nStarting batch $([Math]::Floor($currentBatch / $batchSize) + 1) of $([Math]::Ceiling($totalCommands / $batchSize))"
        
        # Start the next batch of processes
        for ($i = 0; $i -lt $batchSize -and ($currentBatch + $i) -lt $totalCommands; $i++) {
            $command = $commands_random[$currentBatch + $i]
            Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$currentPath'; Write-Host 'Running: $command'; echo y | $command"
            Start-Sleep -Seconds 2
        }
        
        # Move to next batch
        $currentBatch += $batchSize
    }
}