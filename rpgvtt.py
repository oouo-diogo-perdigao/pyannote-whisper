import os
import subprocess
import time
import queue
import argparse
import numpy as np
import torch
import pickle
import threading
from tqdm import tqdm
from pathlib import Path
from tabulate import tabulate
from datetime import datetime
from pynput import keyboard
from typing import Literal, cast, List, Union
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
from threading import Thread

# Carrega as variáveis do arquivo .env
load_dotenv()

# todo organizar codigo separando melhor as classes e funções


# region (collapsed) Thread Progress Bar para o whisper
class ProgressListener:
    def on_progress(self, current: Union[int, float], total: Union[int, float]):
        pass

    def on_finished(self):
        pass


class ProgressListenerHandle:
    def __init__(self, listener: ProgressListener):
        self.listener = listener

    def __enter__(self):
        register_thread_local_progress_listener(self.listener)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unregister_thread_local_progress_listener(self.listener)
        if exc_type is None:
            self.listener.on_finished()


class _CustomProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        # Remove argumentos não reconhecidos
        kwargs.pop("disable", None)
        super().__init__(*args, **kwargs)
        self._current = self.n
        self.total = kwargs.get(
            "total", 100
        )  # Valor padrão para casos sem total definido

    def update(self, n=1):
        super().update(n)
        self._current += n
        listeners = _get_thread_local_listeners()
        for listener in listeners:
            # Normaliza o progresso para porcentagem
            progress_percent = (self._current / self.total) * 100 if self.total else 0
            listener.on_progress(progress_percent, 100)


_thread_local = threading.local()


def _get_thread_local_listeners():
    if not hasattr(_thread_local, "listeners"):
        _thread_local.listeners = []
    return _thread_local.listeners


_hooked = False


def init_progress_hook():
    global _hooked
    if not _hooked:
        import whisper.transcribe

        # Substitui a referência direta da função tqdm no módulo
        whisper.transcribe.tqdm = _CustomProgressBar
        _hooked = True


def register_thread_local_progress_listener(progress_listener: ProgressListener):
    init_progress_hook()
    listeners = _get_thread_local_listeners()
    listeners.append(progress_listener)


def unregister_thread_local_progress_listener(progress_listener: ProgressListener):
    listeners = _get_thread_local_listeners()
    if progress_listener in listeners:
        listeners.remove(progress_listener)


class TqdmProgressListener(ProgressListener):
    def __init__(self, desc: str):
        self.pbar = tqdm(
            desc=desc,
            unit="%",
            ncols=80,
            leave=False,
            bar_format="{l_bar}{bar}| {n:.0f}%",
        )

    def on_progress(self, current: float, total: float):
        self.pbar.n = current
        self.pbar.total = total
        self.pbar.refresh()

    def on_finished(self):
        self.pbar.n = 100
        self.pbar.close()


# endregion

# region (collapsed) pyanoote
from pyannote.core import Segment, Annotation, Timeline


def get_text_with_timestamp(transcribe_res):
    timestamp_texts = []
    for item in transcribe_res["segments"]:
        start = item["start"]
        end = item["end"]
        text = item["text"]
        timestamp_texts.append((Segment(start, end), text))
    return timestamp_texts


def add_speaker_info_to_text(timestamp_texts, ann):
    spk_text = []
    for seg, text in timestamp_texts:
        spk = ann.crop(seg).argmax()
        spk_text.append((seg, spk, text))
    return spk_text


def merge_cache(text_cache):
    sentence = "".join([item[-1] for item in text_cache])
    spk = text_cache[0][1]
    start = text_cache[0][0].start
    end = text_cache[-1][0].end
    return Segment(start, end), spk, sentence


PUNC_SENT_END = [".", "?", "!"]


def merge_sentence(spk_text):
    merged_spk_text = []
    pre_spk = None
    text_cache = []
    for seg, spk, text in spk_text:
        if spk != pre_spk and pre_spk is not None and len(text_cache) > 0:
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = [(seg, spk, text)]
            pre_spk = spk

        elif text and len(text) > 0 and text[-1] in PUNC_SENT_END:
            text_cache.append((seg, spk, text))
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = []
            pre_spk = spk
        else:
            text_cache.append((seg, spk, text))
            pre_spk = spk
    if len(text_cache) > 0:
        merged_spk_text.append(merge_cache(text_cache))
    return merged_spk_text


# endregion


# region (collapsed) minha classe de controle
class VttSpkGenerator:
    def __init__(
        self,
        auth_token,
        language=None,
        model="medium",
        output_format: Literal["txt", "vtt", "srt"] = "vtt",
        device="cuda",
        threads=0,
        whisper=True,
        pyannote=True,
    ):
        # uso interno
        self.skip = False
        self.listener = None
        self.process = None
        self.whisper = whisper
        self.pyannote = pyannote

        # parametros de pacotes
        self.language = language
        self.output_format = output_format

        if threads > 0:
            torch.set_num_threads(threads)

        # Inicia o carregamento dos modelos em threads separadas
        if whisper:
            thread_whisper = Thread(
                target=self.carregar_modelo_whisper, args=(model, device)
            )
        if pyannote:
            thread_pyannote = Thread(
                target=self.carregar_modelo_pyannote, args=(auth_token,)
            )

        if whisper:
            thread_whisper.start()
        if pyannote:
            thread_pyannote.start()

        # Aguarda a conclusão das threads
        if whisper:
            thread_whisper.join()
        if pyannote:
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
        try:
            if not self.whisper:
                raise ValueError("Whisper não está habilitado")
            os.makedirs(mp3_path.parent, exist_ok=True)

            # Configura a barra de progresso
            progress_listener = TqdmProgressListener(
                total=100,
                desc=f"Transcrevendo {mp3_path.parent}/{mp3_path.name} para {self.output_format.upper()}",
            )

            with ProgressListenerHandle(progress_listener):
                transcribeResult = transcribe(
                    self.model,
                    str(mp3_path),
                    verbose=False,  # Desativa o progresso padrão
                    task="transcribe",
                    language=self.language,
                )

            # Resto do código de salvamento...
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

            return transcribeResult

        except subprocess.CalledProcessError as e:
            print(f"\n❌ Erro na transcrição: {str(e)}")
            return None

    # todo ajustar a diarização
    def processar_spk(self, mp3_path, transcribeResult):
        try:
            if not self.whisper:
                raise ValueError("Whisper não está habilitado")
            print(f"\n{'='*40}")
            print(f"\n🔨 Iniciando diarização... .spk")
            print(f"📂 Pasta: {mp3_path.parent}")
            print(f"🎧 Nome: {mp3_path.name}")
            print(
                f"⏰ Modificado: {datetime.fromtimestamp(os.path.getmtime(mp3_path))}"
            )
            print(f"{'='*40}")

            audio_basename = os.path.basename(mp3_path)
            diarization_result = self.pipeline(mp3_path)

            timestamp_texts = get_text_with_timestamp(transcribeResult)
            spk_text = add_speaker_info_to_text(timestamp_texts, diarization_result)
            res = merge_sentence(spk_text)

            with open(
                os.path.join(mp3_path.parent, audio_basename + ".spk"),
                "w",
                encoding="utf-8",
            ) as file:
                for seg, spk, sentence in res:
                    line = f"{seg.start:.2f} {seg.end:.2f} {spk} {sentence}\n"
                    file.write(line)

            print(f"\n{'='*40}")
            print(f"\n🔨 Diarização Finalizada. .spk")
            print(f"📂 Pasta: {mp3_path.parent}")
            print(f"🎧 Nome: {mp3_path.name}")
            print(
                f"⏰ Modificado: {datetime.fromtimestamp(os.path.getmtime(mp3_path))}"
            )
            print(f"\n✅ {self.output_format.upper()} criado: {mp3_path.name}")
            print(f"{'='*40}")

        except subprocess.CalledProcessError as e:
            print(f"\n❌ Erro na Diarização: {str(e)}")
            return None


# endregion


# region (collapsed) Funções auxiliares locais
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


def salvar_objeto(objeto, caminho_arquivo):
    """Salva um objeto Python em um arquivo .txt usando pickle."""
    with open(caminho_arquivo, "wb") as arquivo:
        pickle.dump(objeto, arquivo)


def carregar_objeto(caminho_arquivo):
    """Carrega um objeto Python de um arquivo .txt usando pickle."""
    with open(caminho_arquivo, "rb") as arquivo:
        return pickle.load(arquivo)


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


# endregion


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
        whisper=args.vtt,
        pyannote=args.spk,
    )

    if (
        ((total - vtt) > 0)
        if not args.spk
        else ((total - spk) > 0 or (total - vtt) > 0)
    ):

        def spk_worker():
            while True:
                [mp3_path, transcribeResult] = spk_queue.get()
                if mp3_path is None:
                    break
                if mp3_path.with_suffix(".process").exists():
                    controller.processar_spk(mp3_path, transcribeResult)
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
                if (
                    mp3_path.with_suffix(".process").exists()
                    and not mp3_path.with_name(f"{mp3_path.stem}.spk").exists()
                ):
                    transcribeResult = carregar_objeto(mp3_path.with_suffix(".process"))
                    spk_queue.put([mp3_path, transcribeResult])
                    print(f"📁 Arquivo {mp3_path.name} adicionado à fila")

        if args.vtt:
            pendentesVtt = [f for f in mp3_files if not f.with_suffix(".vtt").exists()]

            with tqdm(
                total=len(pendentesVtt), desc="Processando arquivos", unit="arquivo"
            ) as main_pbar:
                for idx, mp3_path in enumerate(pendentesVtt, 1):
                    print(f"📁 Arquivo {idx}/{len(pendentesVtt)}")

                    transcribeResult = controller.processar_vtt(mp3_path)
                    if transcribeResult:
                        salvar_objeto(
                            transcribeResult, mp3_path.with_suffix(".process")
                        )
                        controller.tocar_30_segundos(mp3_path)
                        spk_queue.put([mp3_path, transcribeResult])
                        print(f"📁 Arquivo {mp3_path.name} adicionado à fila")

                    main_pbar.update(1)

        if args.spk:
            # Aguarda a conclusão da thread_spk antes de prosseguir
            thread_spk.join()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Execução interrompida pelo usuário!")
