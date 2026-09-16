# run_overnight.ps1
# ------------------
# Roda o pipeline COMPLETO de descoberta de estrategias, do zero:
#   correlator (tipos 1+2+3+4) -> backtester (exp1-4) -> walk_forward -> portfolio
#
# Diferente do run_daily.ps1 (que so atualiza sinais com estrategias ja
# aprovadas), este script REDESCOBRE tudo. Leva horas - rodar overnight.
#
# Cada etapa so roda se a anterior terminou com exit code 0, para nao
# gerar resultado misturando dados novos com resultados velhos de uma
# etapa que falhou no meio.
#
# Log completo em reports\run_overnight.log
#
# NOTA: manter este arquivo em ASCII puro. O PowerShell 5.1 le .ps1 como
# ANSI por padrao; acentos e travessoes gravados em UTF-8 viram lixo e
# quebram o parser.

$BASE = "C:\Users\davpl\OneDrive\Documents\PROJETO"
$LOG  = Join-Path $BASE "reports\run_overnight.log"
Set-Location $BASE

function Write-Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LOG -Value "[$ts] $msg" -Encoding utf8
}

Write-Log ""
Write-Log "#########################################################"
Write-Log "RUN OVERNIGHT - pipeline completo de descoberta"
Write-Log "Baseline salvo em reports\baseline_2026-09-01\"
Write-Log "#########################################################"

$etapas = @(
    @{ nome = "1/4 CORRELATOR (tipos 1,2,3,4 - etapa mais longa)"; script = "analysis\correlator.py" },
    @{ nome = "2/4 BACKTESTER (exp1, exp2, exp3, exp4)";           script = "backtest\backtester.py" },
    @{ nome = "3/4 WALK-FORWARD (janela unica + deslizante)";      script = "analysis\walk_forward.py" },
    @{ nome = "4/4 PORTFOLIO vs CDI (sizing + regime + grafico)";  script = "analysis\portfolio_vs_cdi.py" }
)

foreach ($e in $etapas) {
    Write-Log "========================================================="
    Write-Log ("INICIANDO: " + $e.nome)
    Write-Log "========================================================="
    $t0 = Get-Date

    python $e.script *>> $LOG
    $code = $LASTEXITCODE

    $dur = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if ($code -ne 0) {
        Write-Log ("FALHOU (exit " + $code + "): " + $e.nome + " apos " + $dur + " min. Pipeline INTERROMPIDO.")
        exit 1
    }
    Write-Log ("OK: " + $e.nome + " - duracao " + $dur + " min")
}

Write-Log ""
Write-Log "#########################################################"
Write-Log "PIPELINE COMPLETO - todas as 4 etapas terminaram OK"
Write-Log "Grafico final: reports\portfolio_vs_cdi.png"
Write-Log "Comparar com:  reports\baseline_2026-09-01\portfolio_vs_cdi.png"
Write-Log "#########################################################"
