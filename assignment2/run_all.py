import os
import subprocess
import glob

def main():
    archive_dir = "archive"
    
    # Trova tutte le sottocartelle numerate in archive e le ordina
    # glob.glob("archive/*/") restituirà percorsi come "archive/001/", "archive/002/", ecc.
    folders = sorted(glob.glob(os.path.join(archive_dir, "*/")))
    print(f"Trovate {len(folders)} cartelle in '{archive_dir}':")

    if not folders:
        print(f"Nessuna cartella trovata in '{archive_dir}'.")
        return

    for folder in folders:
        # Costruisce il path per la cartella della fotocamera
        camera_dir = os.path.join(folder, "camera/front_camera")
        
        if os.path.isdir(camera_dir):
            print(f"\n{'='*50}")
            print(f"Avvio elaborazione per: {folder}")
            print(f"{'='*50}")
            
            # Lancia il tuo script run_gold.py passandogli il path della camera
            # Usa python3 o python a seconda di come avvii i tuoi script di solito
            subprocess.run(["python3", "run_gold.py", camera_dir])
            
            # Quando run_gold.py termina (dopo che chiudi la sua finestra),
            # lo script si mette in pausa e aspetta un tuo comando
            try:
                input("\n👉 Premi INVIO per passare al prossimo video, o premi Ctrl+C per interrompere tutto... ")
            except KeyboardInterrupt:
                print("\nEsecuzione interrotta dall'utente.")
                break
        else:
            print(f"⚠️ Cartella 'camera' non trovata in {folder}, salto...")

if __name__ == "__main__":
    main()