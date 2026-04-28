param(
    [int]$MinFreeGB = 8
)

$ErrorActionPreference = "Stop"

function Get-FreeSpaceGB {
    $drive = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'"
    if (-not $drive) {
        throw "Could not read C: drive information."
    }
    return [math]::Round(($drive.FreeSpace / 1GB), 2)
}

function Ensure-Docker {
    try {
        docker version *> $null
    }
    catch {
        throw "Docker is not available. Start Docker Desktop and try again."
    }
}

Write-Host "Checking Docker availability..."
Ensure-Docker

$freeGB = Get-FreeSpaceGB
Write-Host "C: free space: $freeGB GB"

if ($freeGB -lt $MinFreeGB) {
    Write-Host ""
    Write-Host "ERROR: Not enough disk space for reliable Docker builds." -ForegroundColor Red
    Write-Host "Required free space: $MinFreeGB GB"
    Write-Host "Current free space : $freeGB GB"
    Write-Host ""
    Write-Host "Run cleanup and retry:"
    Write-Host "  docker builder prune -a -f"
    Write-Host "  docker image prune -a -f"
    Write-Host "  docker container prune -f"
    Write-Host "  docker volume prune -f"
    exit 1
}

Write-Host "Building images..."
docker compose -f docker-compose.yml build --progress=plain

Write-Host "Starting services..."
docker compose -f docker-compose.yml up -d

Write-Host ""
Write-Host "Done. Services are running."
Write-Host "Use: docker compose -f docker-compose.yml ps"
