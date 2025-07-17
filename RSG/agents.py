import openai
import json
import time
from pathlib import Path
import io
import base64
import requests
import spacy
import os
from PIL import Image

openai.api_key = os.environ['OPENAI_API_KEY']

class ChatGPT4Agent(object):

    def __init__(self, root_path, control=False):
        self.model = "gpt-4" 
        self.save_prompt = True
        self.scene_num = 0
        if control:
            self.base_content = "Please generate scene description based on the given information:"
        else:
            self.base_content = "Please generate next scene based on the given scene/scenes information:"
        self.content = self.base_content
        self.root_path = root_path

    def encode_image_pil(self, image):
        with io.BytesIO() as buffer:
            image.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        
    def write_json(self, output, save_dir=None):
        if save_dir is None:
            save_dir = Path(self.root_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            output['background'][0] = self.generate_keywords(output['background'][0])
            with open(save_dir / 'scene_{}.json'.format(str(self.scene_num).zfill(2)), "w") as json_file:
                json.dump(output, json_file, indent=4)
        except Exception as e:
            pass
        return
    
    def write_all_content(self, save_dir=None):
        if save_dir is None:
            save_dir = Path(self.root_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / 'all_content.txt', "w") as f:
            f.write(self.content)
        return

    def detect_iss(self, image):
            
            original_api_key = openai.api_key

            base64_image = self.encode_image_pil(image)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai.api_key}"
            }

            prompt_text = (
                "Please carefully analyze the image. "
                "Identify the most visually unusual or problematic region in the image. (especially Fragmented object)"
                "If you find any such issues, describe the most severe and obvious issue in a single sentence. If not, just say 'No issues found' and describe the image."
            )

            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt_text
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 300
            }

            has_issue = False
            issue_description = ""

            for i in range(100):
                try:
                    response = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=10
                    )
                    result_text = response.json()["choices"][0]["message"]["content"].strip()
                    print(f"Response: {result_text}")
                    if "no issues found" in result_text.lower():
                        has_issue = False
                        issue_description = ""
                    else:
                        has_issue = True
                        issue_description = result_text
                    break
                except Exception as e:
                    print("Request error. Retrying...")
                    time.sleep(1)
                    continue

            # 还原api_key
            openai.api_key = original_api_key

            return has_issue, issue_description
        
    def check_image_issue_with_window(self, image, issue_description):
            
        import openai
        import base64
        import requests
        import time
        import json

        def encode_image_pil(image):
            import io
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

        base64_image = encode_image_pil(image)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }
        prompt_text = (
            f"The user suspects the following issue in the image: \"{issue_description}\". "
            "If the issue is present, return the bounding box of the affected area and a short positive prompt "
            "describing how the region should ideally look after fixing. "
            "Use the following JSON format, and respond with only the JSON object, without any markdown formatting or commentary:\n"
            "{\n"
            "  \"x_min\": <int>,\n"
            "  \"x_max\": <int>,\n"
            "  \"y_min\": <int>,\n"
            "  \"y_max\": <int>,\n"
            "  \"prompt\": \"<short description of what the region should be changed to>\"\n"
            "}\n"
            "Coordinate definition: x refers to horizontal axis (left to right), y refers to vertical axis (top to bottom). "
            "The point (x_min, y_min) is the top-left corner, and (x_max, y_max) is the bottom-right corner of the region. "
            "The bounding box should not cover more than one fourth of the image area. "
            "If there is no issue, respond with \"No\" only."
        )
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 200
        }

        for _ in range(50):
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=10
                )
                content = response.json()["choices"][0]["message"]["content"].strip()
                if content.lower() == "no":
                    return False, None

                try:
                    result = json.loads(content)
                    if all(k in result for k in ("x_min", "x_max", "y_min", "y_max", "prompt")):
                        return True, result
                    else:
                        print(f"Incomplete result keys: {result}")
                except json.JSONDecodeError:
                    print(f"JSON parsing failed: {content}")

                time.sleep(1)

            except Exception as e:
                print(f"Request error ({str(e)}), retrying...")
                time.sleep(1)

        raise RuntimeError("Failed to get a valid window and prompt after multiple attempts.")