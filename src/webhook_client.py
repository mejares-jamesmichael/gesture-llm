import requests
import json


class WebhookClient:
    def __init__(self, url, session_id=None):
        self.url = url
        self.session_id = session_id
        self.debug_mode = False  # Set to True to skip actual webhook calls
    
    def send(self, text):
        if self.debug_mode:
            print(f"[DEBUG] Would send: '{text}' (session: {self.session_id})")
            return {'output': f'[DEBUG MODE] Received: {text}'}
        
        payload = {
            'chat_input': text,
            'sessionId': self.session_id  # Include session ID for conversation context
        }
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        try:
            print(f"Sending to webhook: {self.url}")
            print(f"Payload: {payload}")
            
            # Try POST first
            response = requests.post(self.url, json=payload, headers=headers, timeout=15)
            
            # If POST fails with 404, try GET with query parameters
            if response.status_code == 404 and "not registered for POST" in response.text:
                print("POST failed, trying GET request...")
                response = requests.get(self.url, params=payload, headers=headers, timeout=15)
            
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text[:200]}")
            response.raise_for_status()
            return self._parse_response(response)
        except requests.exceptions.RequestException as e:
            print(f"Error sending to webhook: {e}")
            return None
    
    def _parse_response(self, response):
        try:
            return response.json()
        except ValueError:
            return {'text': response.text}
