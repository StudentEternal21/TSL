import requests

tests = [
    {
        "path": r"data/audio_speech/synthetic/war_001.wav",
        "expected": "Paano kun diri bukad-bukad it' igsul-ot, a-absenan na la kamo?"
    },
    {
        "path": r"data/audio_speech/synthetic/war_002.wav",
        "expected": "Tas pag fourth year liwat an ak goal pag-gain ba hin maupay nga grado para ak makasulod hit eskwelahan nga ak' a-apply-yan."
    },
    {
        "path": r"data/audio_speech/synthetic/war_003.wav",
        "expected": "A-apply-yan ko ha Gaisano kay hiring yana."
    },
    {
        "path": r"data/audio_speech/synthetic/war_004.wav",
        "expected": "Ano an imo a-apply-yan?"
    },
    {
        "path": r"data/audio_speech/synthetic/war_005.wav",
        "expected": "Pero waray igsumat kan Kim kun ano an resulta kahuman niya magpakonsulta kanina han iya sakit nga waray na niya aabata, tigda la hiya gin-atake."
    }
]

def test_transcribe():
    url = "http://127.0.0.1:5000/transcribe"
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
