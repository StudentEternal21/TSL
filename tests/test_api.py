import requests

tests = [
    {
        "path": r"data/audio_speech/synthetic/war_001.wav",
        "expected": "Diri ko aram."
    },
    {
        "path": r"data/audio_speech/synthetic/war_002.wav",
        "expected": "Kinaon ka na."
    },
    {
        "path": r"data/audio_speech/synthetic/war_003.wav",
        "expected": "Ngain ka makadto."
    },
    {
        "path": r"data/audio_speech/synthetic/war_004.wav",
        "expected": "Bisan ano pa kamapasensyahon hit usa nga tawo, may-ada gid hiya limit lalo na kun pirme nala gina-take for granted."
    },
    {
        "path": r"data/audio_speech/synthetic/war_005.wav",
        "expected": "Makaruruyag unta mag-pursue hiton nga career ngem makaharadlok man gud kun diri ka mag-succeed."
    },
    {
        "path": r"data/audio_speech/synthetic/war_006.wav",
        "expected": "Naruruyag ako hiton imo system setup, pero kailangan ta pa ig-align an core features para hiton aton presentation."
    },
    {
        "path": r"data/audio_speech/synthetic/war_007.wav",
        "expected": "Ayaw sige ka-stress hit traffic, ma-abot man gihapon kita didto."
    },
    {
        "path": r"data/audio_speech/synthetic/war_008.wav",
        "expected": "Mag-gi-gym pa ako niyan kay damo la gihapon an akon dapat i-burn nga calories."
    },
    {
        "path": r"data/audio_speech/synthetic/war_009.wav",
        "expected": "Paki-send nala hit file hiton project ta kay i-re-review-hon ko yana nga gabi."
    }
]

def test_transcribe():
    url = "https://36ea-2001-4451-1310-5b00-b1a9-8f35-b714-28df.ngrok-free.app/transcribe"
    dialect = "war"
    
    for t in tests:
        audio_path = t["path"]
        expected = t["expected"]
        
        print("\n" + "="*80)
        print(f"Sending POST request to {url}...")
        print(f"Audio file: {audio_path}")
        print(f"Dialect: {dialect}")
        print(f"Expected:   {expected}")
        
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
