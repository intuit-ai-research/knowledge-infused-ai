import logging
import os
import time

import openai


class LLMConnector:
    max_retries = 5
    wait_seconds = 5
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    @classmethod
    async def get_response(
        cls,
        messages,
        model="gpt-3.5-turbo",
    ):

        times = 0
        while times < cls.max_retries:
            try:
                response = await cls.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_tokens=800,
                )
                cls.logger.debug(response)
                return response.choices[0].message.content
            except Exception as e:
                cls.logger.error(f"Error: {e}")
                times += 1
                time.sleep(cls.wait_seconds)
