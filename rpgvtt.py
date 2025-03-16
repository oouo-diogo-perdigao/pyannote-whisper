import os
import subprocess
import argparse
from pathlib import Path
from tabulate import tabulate
import time
from datetime import datetime
from pynput import keyboard


def listar_arquivos_mp3(pasta_base):
    mp3_files = []
    for root, _, files in os.walk(pasta_base):
        for file in files:
            if file.lower().endswith(".mp3"):
                path = Path(root) / file
                mp3_files.append(path)

    mp3_files.sort(key=lambda x: x.name.lower(), reverse=True)
    return mp3_files


class AudioController:
    def __init__(self):
        self.skip = False
        self.listener = None
        self.process = None

    def on_press(self, key):
        try:
            if key.char.lower() == "s":
                self.skip = True
                print("\n⏭ Pressionou S - Pulando para próximo arquivo...")
        except AttributeError:
            pass

    def tocar_30_segundos(self, audio_path):
        self.skip = False
        try:
            # Inicia a reprodução
            self.process = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-t", "30", str(audio_path)]
            )

            # Configura o listener de teclado
            self.listener = keyboard.Listener(on_press=self.on_press)
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

    def esperar_ou_pular(self):
        self.skip = False
        print(f"\n⏳ Aguardando 30s (pressione S para pular)...")
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

        start_time = time.time()
        while time.time() - start_time < 30:
            if self.skip:
                break
            time.sleep(0.1)

        if self.listener:
            self.listener.stop()
            self.listener = None


def processar_arquivos(arquivos, test_mode=False):
    controller = AudioController()

    for idx, mp3_path in enumerate(arquivos, 1):
        vtt_path = mp3_path.with_suffix(".vtt")

        print(f"\n{'='*40}")
        print(f"📁 Arquivo {idx}/{len(arquivos)}")
        print(f"🎧 Nome: {mp3_path.name}")
        print(f"📂 Pasta: {mp3_path.parent}")
        print(f"⏰ Modificado: {datetime.fromtimestamp(os.path.getmtime(mp3_path))}")
        print(f"{'='*40}")

        if not test_mode and not vtt_path.exists():
            comando = [
                "python",
                "-m",
                "pyannote_whisper.cli.transcribe",
                str(mp3_path),
                "--model",
                "medium",
                "--diarization",
                "True",
                "--task",
                "transcribe",
                "--language",
                "pt",
                "--device",
                "cuda",
                "--output_format",
                "VTT",
            ]

            try:
                print("\n🔨 Processando transcrição...")
                subprocess.run(comando, check=True)
                print(f"\n✅ VTT criado: {vtt_path.name}")
            except subprocess.CalledProcessError as e:
                print(f"\n❌ Falha no processamento: {str(e)}")
                continue

        print("\n🔊 Reproduzindo 30 segundos do áudio...")
        controller.tocar_30_segundos(mp3_path)

        if idx < len(arquivos):
            controller.esperar_ou_pular()


def main():
    parser = argparse.ArgumentParser(description="Processador de Áudio RPG")
    parser.add_argument("pasta_base", help="Diretório raiz com arquivos MP3")
    parser.add_argument(
        "--teste", action="store_true", help="Modo teste: só reproduz áudios"
    )
    args = parser.parse_args()

    mp3_files = listar_arquivos_mp3(args.pasta_base)
    status = [
        [f.name, str(f.parent), "Sim" if f.with_suffix(".vtt").exists() else "Não"]
        for f in mp3_files
    ]

    print("\n📊 Status dos Arquivos:")
    print(
        tabulate(
            status,
            headers=["Arquivo", "Pasta", "VTT Existe"],
            tablefmt="grid",
            showindex=True,
        )
    )

    total = len(status)
    completos = sum(1 for item in status if item[2] == "Sim")
    print(
        f"\n🔢 Total: {total} | ✅ Concluídos: {completos} | ⏳ Pendentes: {total - completos}"
    )

    if not args.teste and (total - completos) > 0:
        resp = input("\n▶️ Iniciar processamento? [Y/n]: ").strip().lower()
        if resp in ("", "y", "yes"):
            pendentes = [f for f in mp3_files if not f.with_suffix(".vtt").exists()]
            processar_arquivos(pendentes)
    elif args.teste:
        print("\n🔧 Modo teste ativado")
        processar_arquivos(mp3_files, test_mode=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Execução interrompida pelo usuário!")
