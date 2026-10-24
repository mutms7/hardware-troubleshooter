param([ValidateSet('none', 'stub')][string]$Backend = 'stub')
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    & uv run --frozen hardware-troubleshooter mcp --backend $Backend
    if ($LASTEXITCODE -ne 0) { throw 'The MCP server exited with an error.' }
}
finally {
    Pop-Location
}
