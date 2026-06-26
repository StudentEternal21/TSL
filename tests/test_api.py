import requests

audio_paths = [
    r"data/audio_speech/WARAY/E_Diri ko aram..wav",
    r"data/audio_speech/WARAY/E_Kinaon ka na.wav",
    r"data/audio_speech/WARAY/E_Ngain ka makadto.wav",
    r"data/audio_speech/WARAY/H_Bisan ano pa kamapasensyahon hit usa nga tawo, may-ada gid hiya limit lalo na kun pirme nala gina-take for granted..wav",
    r"data/audio_speech/WARAY/H_Makaruruyag unta mag-pursue hiton nga career ngem makaharadlok man gud kun diri ka mag-succeed..wav",
    r"data/audio_speech/WARAY/H_Naruruyag ako hiton imo system setup, pero kailangan ta pa ig-align an core features para hiton aton presentation..wav",
    r"data/audio_speech/WARAY/M_Ayaw sige ka-stress hit traffic, ma-abot man gihapon kita didto..wav",
    r"data/audio_speech/WARAY/M_Mag-gi-gym pa ako niyan kay damo la gihapon an akon dapat i-burn nga calories..wav",
    r"data/audio_speech/WARAY/M_Paki-send nala hit file hiton project ta kay i-re-review-hon ko yana nga gabi..wav"
]

def test_transcribe():
    url = "http://127.0.0.1:5000/transcribe"
    dialect = "war"
    
    for audio_path in audio_paths:
        print("\n" + "="*80)
        print(f"Sending POST request to {url}...")
        print(f"Audio file: {audio_path}")
        print(f"Dialect: {dialect}")
        
        try:
            with open(audio_path, "rb") as f:
                files = {"audio": f}
                data = {"dialect": dialect}
                r = requests.post(url, files=files, data=data)
                
            print(f"Response Status Code: {r.status_code}")
            if r.status_code == 200:
                print("Response JSON:")
                import json
                print(json.dumps(r.json(), indent=2))
            else:
                print("Response Text:")
                print(r.text)
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_transcribe()
