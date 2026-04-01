from pyngrok import ngrok

token = input("Paste your ngrok token: ").strip()
ngrok.set_auth_token(token)

public_url = ngrok.connect(5000)
print("\n" + "="*50)
print(f"Your app is live at: {public_url}")
print("Open this link on your mobile!")
print("="*50)
print("\nPress Ctrl+C to stop...")

try:
    input()
except KeyboardInterrupt:
    ngrok.kill()
