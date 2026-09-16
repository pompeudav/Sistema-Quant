"""
baixar_cvm.py
Baixa os arquivos mensais de cotas de fundos da CVM.
Uso: python baixar_cvm.py
"""
import os
import time
import zipfile
import requests
from datetime import date

PASTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw", "cvm")
BASE_URL = "https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    "Referer": "https://dados.cvm.gov.br/",
}

def baixar(url, destino):
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code == 200 and len(r.content) > 500:
        with open(destino, "wb") as f:
            f.write(r.content)
        return True
    return False

def main():
    os.makedirs(PASTA, exist_ok=True)
    hoje = date.today()
    ok, falhas = 0, []

    # Anos históricos em ZIP (até 2020)
    for ano in range(2020, 2021):
        arquivo = f"inf_diario_fi_{ano}.zip"
        destino = os.path.join(PASTA, arquivo)
        if os.path.exists(destino):
            print(f"  já existe: {arquivo}")
            ok += 1
            continue
        url = f"{BASE_URL}/HIST/{arquivo}"
        print(f"  {arquivo}...", end=" ", flush=True)
        if baixar(url, destino):
            print(f"OK ({os.path.getsize(destino)//1024} KB)")
            ok += 1
        else:
            print("FALHA")
            falhas.append(arquivo)
        time.sleep(0.2)

    # Mensais a partir de 2021
    for ano in range(2021, hoje.year + 1):
        mes_fim = hoje.month if ano == hoje.year else 12
        for mes in range(1, mes_fim + 1):
            arquivo = f"inf_diario_fi_{ano}{mes:02d}.csv"
            destino = os.path.join(PASTA, arquivo)
            if os.path.exists(destino):
                print(f"  já existe: {arquivo}")
                ok += 1
                continue
            url = f"{BASE_URL}/{arquivo}"
            print(f"  {arquivo}...", end=" ", flush=True)
            if baixar(url, destino):
                print(f"OK ({os.path.getsize(destino)//1024} KB)")
                ok += 1
            else:
                print("FALHA (provavelmente ainda não publicado)")
                falhas.append(arquivo)
            time.sleep(0.15)

    print(f"\nConcluído: {ok} arquivos OK | {len(falhas)} falhas")
    if falhas:
        print("Falhas:", falhas)

if __name__ == "__main__":
    main()
