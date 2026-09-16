# run_daily.ps1
# -------------
# Runner diario do sistema quant. Agendado no Windows Task Scheduler como
# a tarefa "QuantPaperTrading" (19h30, dias uteis).
#
# O que este script faz:
#   1. Atualiza dados de mercado (data\collector.py)
#   2. Regenera monitor_status.csv (analysis\gerar_monitor.py)
#   3. Paper trading: registra sinais e fecha posicoes vencidas
#   4. Salva log com timestamp em reports\run_daily.log
#
# Quando integrar API da corretora (Fase 5.5+):
#   adicionar chamada a executar_ordens.py depois do paper_trading.py
#
# ATENCAO - manter este arquivo em ASCII PURO, sem acentos nem tracos
# longos. O PowerShell 5.1 le .ps1 como ANSI por padrao; caracteres
# gravados em UTF-8 viram lixo e quebram o parser. A versao anterior
# deste arquivo estava assim e nunca chegou a rodar: falhava no parse
# antes de executar qualquer linha.

$PROJETO = "C:\Users\davpl\OneDrive\Documents\PROJETO"
$PYTHON  = "$PROJETO\data\.venv\Scripts\python.exe"
$LOG     = "$PROJETO\reports\run_daily.log"

function Write-Etapa {
    param([string]$mensagem)
    $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    $linha = "[$ts] $mensagem"
    Write-Host $linha
    Add-Content -Path $LOG -Value $linha -Encoding utf8
}

function Invoke-Etapa {
    param([string]$descricao, [string[]]$argumentos)
    Write-Etapa "INICIO: $descricao"
    $inicio = Get-Date
    & $PYTHON @argumentos
    $code = $LASTEXITCODE
    $seg = [int]((Get-Date) - $inicio).TotalSeconds
    if ($code -eq 0) {
        Write-Etapa "OK    : $descricao ($seg s)"
        return $true
    }
    Write-Etapa "ERRO  : $descricao ($seg s) - exit code $code"
    return $false
}

$data_hoje = (Get-Date).ToString("yyyy-MM-dd")
Write-Etapa "======================================================"
Write-Etapa "RUN DAILY - $data_hoje"
Write-Etapa "======================================================"

$dia_semana = (Get-Date).DayOfWeek
if ($dia_semana -eq "Saturday" -or $dia_semana -eq "Sunday") {
    Write-Etapa "Final de semana - pulando execucao."
    exit 0
}

# Guarda contra conflito com processos pesados do proprio projeto.
#
# Historico desta guarda (importante para nao repetir o erro):
#   - 02/09: a tarefa rodou junto com o walk_forward_descoberta.py e 4 das
#     5 etapas morreram sem memoria, silenciosamente (so "exit code 1").
#   - 03/09 a 12/09: a primeira versao desta guarda exigia 3 GB livres.
#     Esta maquina opera normalmente com 1-2.5 GB livres (Edge, Chrome,
#     VS Code, Defender), entao a tarefa ABORTOU TODOS OS DIAS por 11 dias
#     seguidos e o paper trading nao registrou um unico pregao. A protecao
#     virou o proprio bloqueio.
#
# O problema real nunca foi RAM baixa em geral - as etapas rodam bem com
# 1.5 GB. O problema era competir com o correlator/walk-forward, que
# sozinhos ocupam varios GB. Entao a guarda agora e especifica: espera se
# um desses processos estiver ativo, e so aborta por RAM se estiver
# criticamente baixa (abaixo de 0.8 GB, onde carregar o retornos.csv
# falha de fato).
$RAM_CRITICA_GB = 0.8
$PESADOS = @("walk_forward_descoberta", "run_overnight", "correlator", "backtester", "walk_forward.py")

function Test-ProcessoPesado {
    $cmds = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty CommandLine
    foreach ($c in $cmds) {
        foreach ($p in $PESADOS) {
            if ($c -and $c -like "*$p*") { return $p }
        }
    }
    return $null
}

$tentativas = 0
while ($tentativas -lt 6) {
    $pesado = Test-ProcessoPesado
    $os = Get-CimInstance Win32_OperatingSystem
    $livre_gb = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
    if (-not $pesado -and $livre_gb -ge $RAM_CRITICA_GB) { break }
    $tentativas++
    if ($pesado) {
        Write-Etapa "Processo pesado ativo ($pesado) - aguardando 10 min (tentativa $tentativas/6)"
    } else {
        Write-Etapa "RAM criticamente baixa ($livre_gb GB) - aguardando 10 min (tentativa $tentativas/6)"
    }
    Start-Sleep -Seconds 600
}
$pesado = Test-ProcessoPesado
$os = Get-CimInstance Win32_OperatingSystem
$livre_gb = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
if ($pesado -or $livre_gb -lt $RAM_CRITICA_GB) {
    Write-Etapa "ABORTADO apos 1h: pesado=$pesado | RAM livre $livre_gb GB."
    Write-Etapa "Rode manualmente quando o processo pesado terminar."
    exit 1
}
Write-Etapa "RAM livre: $livre_gb GB - prosseguindo"

# A ordem das etapas importa:
#   dados novos -> avalia degradacao -> regenera monitor (ja sem as
#   suspensas) -> recalcula sizes das que sobraram -> opera.
# Trocar a ordem faria o paper trading operar com pesos calculados para
# um conjunto de estrategias diferente do que esta ativo.

# 1. Atualiza dados de mercado
$ok = Invoke-Etapa "collector.py (atualiza retornos.csv e cdi.csv)" @(
    "$PROJETO\data\collector.py"
)
if (-not $ok) {
    Write-Etapa "Aviso: collector falhou. Seguindo com os dados anteriores."
}

# 2. Monitor de degradacao: desliga o que parou de funcionar, usando o
# resultado real do paper trading (nao re-roda backtest).
$ok = Invoke-Etapa "monitor_degradacao.py (suspende estrategias degradadas)" @(
    "$PROJETO\analysis\monitor_degradacao.py"
)
if (-not $ok) {
    Write-Etapa "Aviso: monitor de degradacao falhou. Seguindo sem novas suspensoes."
}

# 3. Regenera monitor_status.csv (respeitando as suspensoes da etapa 2)
# --enriquecer calcula metricas recentes (30/60/90d) e demora ~5-10 min;
# roda so as sextas. Nos demais dias, versao rapida.
if ($dia_semana -eq "Friday") {
    $ok = Invoke-Etapa "gerar_monitor.py --enriquecer (semanal)" @(
        "$PROJETO\analysis\gerar_monitor.py", "--enriquecer"
    )
} else {
    $ok = Invoke-Etapa "gerar_monitor.py (rapido)" @(
        "$PROJETO\analysis\gerar_monitor.py"
    )
}

# 4. Recalcula os tamanhos de posicao com as estrategias que sobraram.
# Sem esta etapa os sizes ficam congelados na data em que foram gerados,
# e o capital continua alocado como se nada tivesse mudado.
$ok = Invoke-Etapa "position_sizing.py (recalcula tamanhos)" @(
    "$PROJETO\analysis\position_sizing.py"
)
if (-not $ok) {
    Write-Etapa "Aviso: position_sizing falhou. Paper trading usara os sizes anteriores."
}

# 5. Paper trading: registra sinais do dia e fecha posicoes vencidas
$ok = Invoke-Etapa "paper_trading.py (sinais + fechamento)" @(
    "$PROJETO\analysis\paper_trading.py"
)
if (-not $ok) {
    Write-Etapa "ERRO no paper trading - verificar manualmente."
}

Write-Etapa "======================================================"
Write-Etapa "CONCLUIDO - $data_hoje"
Write-Etapa "======================================================"
Write-Etapa ""

# Mantem apenas os ultimos 90 dias de log
$corte = (Get-Date).AddDays(-90).ToString("yyyy-MM-dd")
$linhas = Get-Content -Path $LOG -Encoding utf8 | Where-Object {
    if ($_ -match "^\[(\d{4}-\d{2}-\d{2})") { $Matches[1] -ge $corte } else { $true }
}
$linhas | Set-Content -Path $LOG -Encoding utf8
