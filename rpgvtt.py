import os
import subprocess
import time
import queue
import argparse
import numpy as np
import torch
import pickle
from pathlib import Path
from tabulate import tabulate
from datetime import datetime
from pynput import keyboard
from typing import Literal, cast
from dotenv import load_dotenv
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
from whisper.transcribe import transcribe
from whisper.utils import (
    WriteSRT,
    WriteTXT,
    WriteVTT,
    optional_int,
    str2bool,
)
from pyannote_whisper.utils import diarize_text, write_to_txt
from threading import Thread

# Carrega as variáveis do arquivo .env
load_dotenv()


def salvar_objeto(objeto, caminho_arquivo):
    """Salva um objeto Python em um arquivo .txt usando pickle."""
    with open(caminho_arquivo, "wb") as arquivo:
        pickle.dump(objeto, arquivo)


def carregar_objeto(caminho_arquivo):
    """Carrega um objeto Python de um arquivo .txt usando pickle."""
    with open(caminho_arquivo, "rb") as arquivo:
        return pickle.load(arquivo)


class VttSpkGenerator:
    def __init__(
        self,
        auth_token,
        language=None,
        model="medium",
        output_format: Literal["txt", "vtt", "srt"] = "vtt",
        device="cuda",
        threads=0,
    ):
        # uso interno
        self.skip = False
        self.listener = None
        self.process = None

        # parametros de pacotes
        self.language = language
        self.output_format = output_format

        if threads > 0:
            torch.set_num_threads(threads)

        # Inicia o carregamento dos modelos em threads separadas
        thread_whisper = Thread(
            target=self.carregar_modelo_whisper, args=(model, device)
        )
        thread_pyannote = Thread(
            target=self.carregar_modelo_pyannote, args=(auth_token,)
        )

        thread_whisper.start()
        thread_pyannote.start()

        # Aguarda a conclusão das threads
        thread_whisper.join()
        thread_pyannote.join()

    def carregar_modelo_whisper(self, model, device):
        from whisper import load_model

        self.model = load_model(model, device)

    def carregar_modelo_pyannote(self, auth_token):
        from pyannote.audio import Pipeline

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization", use_auth_token=auth_token
        )

    def tocar_30_segundos(self, audio_path):
        print("\n🔊 {audio_path} (pressione qualquer tecla para pular)")

        def on_press(self, key):
            self.skip = True

        self.skip = False
        try:
            # Inicia a reprodução
            self.process = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-t", "30", str(audio_path)]
            )

            # Configura o listener de teclado
            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()

            # Aguarda término ou interrupção
            start_time = time.time()
            while time.time() - start_time < 30:
                if self.skip:
                    break
                time.sleep(0.1)

            # Finaliza a reprodução
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait()

        except Exception as e:
            print(f"\n❌ Erro ao reproduzir o áudio: {str(e)}")
        finally:
            if self.listener:
                self.listener.stop()
                self.listener = None

    def processar_vtt(self, mp3_path):
        print(f"\n{'='*40}")
        print(f"\n🔨 Iniciando transcrição... {self.output_format.upper()}")
        print(f"📂 Pasta: {mp3_path.parent}")
        print(f"🎧 Nome: {mp3_path.name}")
        print(f"⏰ Modificado: {datetime.fromtimestamp(os.path.getmtime(mp3_path))}")
        print(f"{'='*40}")

        try:
            # cria a pasta de saída se não existir
            os.makedirs(mp3_path.parent, exist_ok=True)

            transcribeResult = transcribe(
                self.model,
                str(mp3_path),
                verbose=True,
                # initial_prompt=None,
                task="transcribe",
                language=self.language,
            )

            audio_basename = os.path.basename(mp3_path)

            if self.output_format == "txt":
                # save TXT
                with open(
                    os.path.join(mp3_path.parent, audio_basename + ".txt"),
                    "w",
                    encoding="utf-8",
                ) as file:
                    WriteTXT(mp3_path.parent).write_result(transcribeResult, file=file)
            elif self.output_format == "vtt":
                # save VTT
                with open(
                    os.path.join(mp3_path.parent, audio_basename + ".vtt"),
                    "w",
                    encoding="utf-8",
                ) as file:
                    WriteVTT(mp3_path.parent).write_result(transcribeResult, file=file)
            elif self.output_format == "srt":
                # save SRT
                with open(
                    os.path.join(mp3_path.parent, audio_basename + ".srt"),
                    "w",
                    encoding="utf-8",
                ) as file:
                    WriteSRT(mp3_path.parent).write_result(transcribeResult, file=file)

            print(f"\n{'='*40}")
            print(f"\n🔨 Transcrição Finalizada. {self.output_format.upper()}")
            print(f"📂 Pasta: {mp3_path.parent}")
            print(f"🎧 Nome: {mp3_path.name}")
            print(
                f"⏰ Modificado: {datetime.fromtimestamp(os.path.getmtime(mp3_path))}"
            )
            print(f"\n✅ {self.output_format.upper()} criado: {mp3_path.name}")
            print(f"{'='*40}")

            return transcribeResult

        except subprocess.CalledProcessError as e:
            print(f"\n❌ Erro na transcrição: {str(e)}")
            return None

    def processar_spk(self, mp3_path, transcribeResult):
        print(f"\n{'='*40}")
        print(f"\n🔨 Iniciando diarização... .spk")
        print(f"📂 Pasta: {mp3_path.parent}")
        print(f"🎧 Nome: {mp3_path.name}")
        print(f"⏰ Modificado: {datetime.fromtimestamp(os.path.getmtime(mp3_path))}")
        print(f"{'='*40}")

        audio_basename = os.path.basename(mp3_path)
        diarization_result = self.pipeline(mp3_path)
        filepath = os.path.join(mp3_path.parent, audio_basename + ".spk")
        res = diarize_text(transcribeResult, diarization_result)
        write_to_txt(res, filepath)

        print(f"\n{'='*40}")
        print(f"\n🔨 Diarização Finalizada. .spk")
        print(f"📂 Pasta: {mp3_path.parent}")
        print(f"🎧 Nome: {mp3_path.name}")
        print(f"⏰ Modificado: {datetime.fromtimestamp(os.path.getmtime(mp3_path))}")
        print(f"\n✅ {self.output_format.upper()} criado: {mp3_path.name}")
        print(f"{'='*40}")


def listar_arquivos_mp3(pasta_base):
    mp3_files = []

    for item in pasta_base:
        path = Path(item)
        if path.is_file() and path.suffix.lower() == ".mp3":
            mp3_files.append(path)
        elif path.is_dir():
            for root, _, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(".mp3"):
                        mp3_path = Path(root) / file
                        mp3_files.append(mp3_path)
        else:
            # Handle case where the path does not exist
            print(f"Warning: '{item}' does not exist and will be skipped.")

    # Remove duplicates and sort Z-A
    mp3_files = list(set(mp3_files))
    mp3_files.sort(key=lambda x: x.name.lower(), reverse=True)

    if not mp3_files:
        raise ValueError("Nenhum arquivo .mp3 encontrado")
    return mp3_files


def contar(mp3_files):
    status = []
    for mp3 in mp3_files:
        # verifica se existe um arquivo .vtt com o nome do mp3 na mesma pasta
        process = mp3.with_name(f"{mp3.stem}.process").exists()
        vtt = mp3.with_suffix(".vtt").exists()
        spk = mp3.with_name(f"{mp3.stem}.spk").exists()
        status.append(
            [
                mp3.name,
                str(mp3.parent),
                "✅" if process else "❌",
                "✅" if vtt else "❌",
                "✅" if spk else "❌",
            ]
        )

    total = len(status)
    process = sum(1 for item in status if item[2] == "✅")
    vtt = sum(1 for item in status if item[2] == "✅")
    spk = sum(1 for item in status if item[4] == "✅")

    print("\n📊 Status dos Arquivos:")
    print(
        tabulate(
            status,
            headers=["Arquivo", "Pasta", "Process", "VTT", "SPK"],
            tablefmt="grid",
            showindex=True,
        )
    )
    print(
        f"""
        ⚙️ process: {process} / {total} {(process/total)*100:.0f}%
        📝 VTT: {vtt} / {total} {(vtt/total)*100:.0f}%
        🔢 SPK: {spk} / {total} {(spk/total)*100:.0f}%
        """
    )

    return total, vtt, spk


def getArgs():
    from whisper import available_models

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Processador de Áudio RPG",
    )
    parser.add_argument(
        "audio", nargs="+", type=str, help="audio file(s) to transcribe or folder path"
    )
    parser.add_argument(
        "--model",
        default="medium",
        choices=available_models(),
        help="name of the Whisper model to use",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to use for PyTorch inference",
    )
    parser.add_argument(
        "--threads",
        type=optional_int,
        default=0,
        help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="pt",
        choices=sorted(LANGUAGES.keys())
        + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
        help="language spoken in the audio, specify None to perform language detection",
    )
    parser.add_argument(
        "--output_format",
        "-f",
        type=str,
        default="vtt",
        choices=["txt", "vtt", "srt"],
        help="format of the output file; if not specified, all available formats will be produced",
    )
    parser.add_argument(
        "--vtt",
        type=str2bool,
        default=True,
        help="Process vtt?",
    )
    parser.add_argument(
        "--spk",
        type=str2bool,
        default=True,
        help="Process spk?",
    )
    return parser.parse_args()


def main():
    args = getArgs()

    mp3_files = listar_arquivos_mp3(args.audio)
    total, vtt, spk = contar(mp3_files)

    controller = VttSpkGenerator(
        auth_token=os.getenv("HF_AUTH_TOKEN"),
        language=args.language,
        model=args.model,
        output_format=args.output_format,
        device=args.device,
        threads=args.threads,
    )

    if (
        ((total - vtt) > 0)
        if not args.spk
        else ((total - spk) > 0 or (total - vtt) > 0)
    ):

        def spk_worker():
            while True:
                mp3 = spk_queue.get()
                if mp3 is None:
                    break
                if mp3.with_suffix(".vtt").exists():
                    controller.processar_spk(mp3)

                spk_queue.task_done()

        thread_spk = Thread(target=spk_worker, daemon=True)

        if args.spk:
            spk_queue = queue.Queue()
            thread_spk.start()

            # primeiro percorre os arquivos .txt salvos de outras iterações e adiciona eles na fila de diarização
            pendentesSpk = [
                f for f in mp3_files if not f.with_name(f"{f.stem}.spk").exists()
            ]
            for mp3_path in pendentesSpk:
                if mp3_path.with_suffix(".txt").exists():
                    # verifica se o .spk desses arquivos já foi criado
                    if not mp3_path.with_name(f"{mp3_path.stem}.spk").exists():
                        transcribeResult = carregar_objeto(
                            mp3_path.with_suffix(".process")
                        )
                        spk_queue.put([mp3_path, transcribeResult])

        if args.vtt:
            pendentesVtt = [f for f in mp3_files if not f.with_suffix(".vtt").exists()]
            for idx, mp3_path in enumerate(pendentesVtt, 1):
                print(f"📁 Arquivo {idx}/{len(pendentesVtt)}")

                transcribeResult = controller.processar_vtt(mp3_path)

                # Salva o retorno do transcribe em um arquivo .txt
                salvar_objeto(transcribeResult, mp3_path.with_suffix(".process"))

                # toca o audio por 30 segundos para um feedback que foi concluido o vtt
                controller.tocar_30_segundos(mp3_path)

                if idx < len(pendentesVtt):
                    ("Proximo arquivo")

                if args.spk:
                    if mp3_path.with_suffix(".vtt").exists():
                        spk_queue.put([mp3_path, transcribeResult])

        if args.spk:
            # Aguarda a conclusão da thread_spk antes de prosseguir
            thread_spk.join()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Execução interrompida pelo usuário!")
