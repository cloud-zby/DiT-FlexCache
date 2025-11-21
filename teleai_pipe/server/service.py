import asyncio
import threading
import time
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
import torch.multiprocessing as mp
from loguru import logger
from .audio_utils import is_base64_audio, save_base64_audio
from .config import server_config
from .image_utils import is_base64_image, save_base64_image
from .schema import TaskRequest, TaskResponse
from .utils import import_function_from_file

mp.set_start_method("spawn", force=True)


class FileService:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.input_image_dir = cache_dir / "inputs" / "imgs"
        self.input_audio_dir = cache_dir / "inputs" / "audios"
        self.output_video_dir = cache_dir / "outputs"

        self._http_client = None
        self._client_lock = asyncio.Lock()

        self.max_retries = 3
        self.retry_delay = 1.0
        self.max_retry_delay = 10.0

        for directory in [
            self.input_image_dir,
            self.output_video_dir,
            self.input_audio_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create a persistent HTTP client with connection pooling."""
        async with self._client_lock:
            if self._http_client is None or self._http_client.is_closed:
                timeout = httpx.Timeout(
                    connect=10.0,
                    read=30.0,
                    write=10.0,
                    pool=5.0,
                )
                limits = httpx.Limits(
                    max_keepalive_connections=5,
                    max_connections=10,
                    keepalive_expiry=30.0,
                )
                self._http_client = httpx.AsyncClient(
                    verify=False, timeout=timeout, limits=limits, follow_redirects=True
                )
            return self._http_client

    async def _download_with_retry(
        self, url: str, max_retries: Optional[int] = None
    ) -> httpx.Response:
        """Download with exponential backoff retry logic."""
        if max_retries is None:
            max_retries = self.max_retries

        last_exception = None

        retry_delay = self.retry_delay

        for attempt in range(max_retries):
            try:
                client = await self._get_http_client()
                response = await client.get(url)

                if response.status_code == 200:
                    return response
                elif response.status_code >= 500:
                    logger.warning(
                        f"Server error {response.status_code} for {url}, attempt {attempt + 1}/{max_retries}"
                    )
                    last_exception = httpx.HTTPStatusError(
                        f"Server returned {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                else:
                    raise httpx.HTTPStatusError(
                        f"Client error {response.status_code}",
                        request=response.request,
                        response=response,
                    )

            except (
                httpx.ConnectError,
                httpx.TimeoutException,
                httpx.NetworkError,
            ) as e:
                logger.warning(
                    f"Connection error for {url}, attempt {attempt + 1}/{max_retries}: {str(e)}"
                )
                last_exception = e
            except httpx.HTTPStatusError as e:
                if e.response and e.response.status_code < 500:
                    raise
                last_exception = e
            except Exception as e:
                logger.error(f"Unexpected error downloading {url}: {str(e)}")
                last_exception = e

            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, self.max_retry_delay)

        error_msg = f"All {max_retries} connection attempts failed for {url}"
        if last_exception:
            error_msg += f": {str(last_exception)}"
        raise httpx.ConnectError(error_msg)

    async def download_image(self, image_url: str) -> Path:
        """Download image with retry logic and proper error handling."""
        try:
            parsed_url = urlparse(image_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid URL format: {image_url}")

            response = await self._download_with_retry(image_url)

            image_name = Path(parsed_url.path).name
            if not image_name:
                image_name = f"{uuid.uuid4()}.jpg"

            image_path = self.input_image_dir / image_name
            image_path.parent.mkdir(parents=True, exist_ok=True)

            with open(image_path, "wb") as f:
                f.write(response.content)

            logger.info(
                f"Successfully downloaded image from {image_url} to {image_path}"
            )
            return image_path

        except httpx.ConnectError as e:
            logger.error(
                f"Connection error downloading image from {image_url}: {str(e)}"
            )
            raise ValueError(f"Failed to connect to {image_url}: {str(e)}")
        except httpx.TimeoutException as e:
            logger.error(f"Timeout downloading image from {image_url}: {str(e)}")
            raise ValueError(f"Download timeout for {image_url}: {str(e)}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error downloading image from {image_url}: {str(e)}")
            raise ValueError(f"HTTP error for {image_url}: {str(e)}")
        except ValueError as e:
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error downloading image from {image_url}: {str(e)}"
            )
            raise ValueError(f"Failed to download image from {image_url}: {str(e)}")

    async def download_audio(self, audio_url: str) -> Path:
        """Download audio with retry logic and proper error handling."""
        try:
            parsed_url = urlparse(audio_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid URL format: {audio_url}")

            response = await self._download_with_retry(audio_url)

            audio_name = Path(parsed_url.path).name
            if not audio_name:
                audio_name = f"{uuid.uuid4()}.mp3"

            audio_path = self.input_audio_dir / audio_name
            audio_path.parent.mkdir(parents=True, exist_ok=True)

            with open(audio_path, "wb") as f:
                f.write(response.content)

            logger.info(
                f"Successfully downloaded audio from {audio_url} to {audio_path}"
            )
            return audio_path

        except httpx.ConnectError as e:
            logger.error(
                f"Connection error downloading audio from {audio_url}: {str(e)}"
            )
            raise ValueError(f"Failed to connect to {audio_url}: {str(e)}")
        except httpx.TimeoutException as e:
            logger.error(f"Timeout downloading audio from {audio_url}: {str(e)}")
            raise ValueError(f"Download timeout for {audio_url}: {str(e)}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error downloading audio from {audio_url}: {str(e)}")
            raise ValueError(f"HTTP error for {audio_url}: {str(e)}")
        except ValueError as e:
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error downloading audio from {audio_url}: {str(e)}"
            )
            raise ValueError(f"Failed to download audio from {audio_url}: {str(e)}")

    def save_uploaded_file(self, file_content: bytes, filename: str) -> Path:
        file_extension = Path(filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = self.input_image_dir / unique_filename

        with open(file_path, "wb") as f:
            f.write(file_content)

        return file_path

    def get_output_path(self, save_video_path: str) -> Path:
        video_path = Path(save_video_path)
        if not video_path.is_absolute():
            return self.output_video_dir / save_video_path
        return video_path

    async def cleanup(self):
        """Cleanup resources including HTTP client."""
        async with self._client_lock:
            if self._http_client and not self._http_client.is_closed:
                await self._http_client.aclose()
                self._http_client = None


class PipelineService:
    def __init__(self):
        self.manager = None
        self.shared_data = None
        self.task_event = None
        self.result_event = None
        self.is_running = False
        self.task = None
        self.pipeline = None
        self.run_ppl_func = None

    def start_pipeline(self, ppl_file, parallelism, task) -> bool:
        if self.is_running:
            logger.warning("Distributed inference service is already running")
            return True
        try:
            self.task = task
            get_pipeline = import_function_from_file(ppl_file, "get_pipeline")
            self.run_ppl_func = import_function_from_file(ppl_file, "run_with_file")
            self.ppl_file = ppl_file
            self.parallelism = parallelism
            # Create shared data structures
            self.manager = mp.Manager()
            self.shared_data = self.manager.dict()
            self.task_event = self.manager.Event()
            self.result_event = self.manager.Event()

            # Initialize shared data
            self.shared_data["current_task"] = None
            self.shared_data["result"] = None
            self.shared_data["stop"] = False
            self.pipeline = get_pipeline(parallelism=parallelism)
            self.is_running = True
            return True

        except Exception as e:
            logger.exception(
                f"Error occurred while starting distributed inference service: {str(e)}"
            )
            self.stop_pipeline_inference()
            return False

    def stop_pipeline_inference(self):
        assert self.task_event, "Task event is not initialized"
        assert self.result_event, "Result event is not initialized"

        if not self.is_running:
            return

        try:
            logger.info(
                f"Stopping {len(self.processes)} distributed inference service processes..."
            )
            del self.pipeline
            self.pipeline = None
        except Exception as e:
            logger.exception(
                f"Error occurred while stopping distributed inference service: {str(e)}"
            )
        finally:
            # Clean up resources
            self.processes = []
            self.manager = None
            self.shared_data = None
            self.task_event = None
            self.result_event = None
            self.is_running = False

    def submit_task(self, task_data: dict) -> bool:
        assert self.task_event, "Task event is not initialized"
        assert self.result_event, "Result event is not initialized"

        if not self.is_running or not self.shared_data:
            logger.error("Distributed inference service is not started")
            return False

        try:
            self.result_event.clear()
            self.shared_data["result"] = None
            self.shared_data["current_task"] = task_data
            self.task_event.set()  # Signal workers
            self.run_ppl_func(self.pipeline, **task_data)
            self.result_event.set()
            self.shared_data["result"] = {
                "task_id": task_data["task_id"],
                "status": "success",
                "save_video_path": task_data.get(
                    "video_path", task_data["save_video_path"]
                ),  # Return original path for API
                "message": "Inference completed",
            }
            logger.info(f"Task {task_data['task_id']} success")
            return True
        except Exception as e:
            logger.error(f"Failed to submit task: {str(e)}")
            return False

    def wait_for_result(
        self, task_id: str, timeout: Optional[int] = None
    ) -> Optional[dict]:
        assert self.task_event, "Task event is not initialized"
        assert self.result_event, "Result event is not initialized"

        if timeout is None:
            timeout = server_config.task_timeout
        if not self.is_running or not self.shared_data:
            return None

        if self.result_event.wait(timeout=timeout):
            result = self.shared_data.get("result")
            if result and result.get("task_id") == task_id:
                self.shared_data["current_task"] = None
                self.task_event.clear()
                return result

        return None

    def wait_for_result_with_stop(
        self, task_id: str, stop_event: threading.Event, timeout: Optional[int] = None
    ) -> Optional[dict]:
        if timeout is None:
            timeout = server_config.task_timeout

        if not self.is_running or not self.shared_data:
            return None

        assert self.task_event, "Task event is not initialized"
        assert self.result_event, "Result event is not initialized"

        start_time = time.time()

        while time.time() - start_time < timeout:
            if stop_event.is_set():
                logger.info(f"Task {task_id} stop event triggered during wait")
                self.shared_data["current_task"] = None
                self.task_event.clear()
                return None

            if self.result_event.wait(timeout=0.5):
                result = self.shared_data.get("result")
                if result and result.get("task_id") == task_id:
                    self.shared_data["current_task"] = None
                    self.task_event.clear()
                    return result

        return None

    def server_metadata(self):
        return {
            "pipeline_file": self.ppl_file,
            "parallelism": self.parallelism,
            "task": self.task,
        }

    def __del__(self):
        if self.pipeline is not None:
            del self.pipeline


class VideoGenerationService:
    def __init__(self, file_service: FileService, inference_service: PipelineService):
        self.file_service = file_service
        self.inference_service = inference_service

    async def generate_video_with_stop_event(
        self, message: TaskRequest, stop_event
    ) -> Optional[TaskResponse]:
        try:
            task_data = message.model_dump()
            if stop_event.is_set():
                logger.info(f"Task {message.task_id} cancelled before processing")
                return None

            async def update_image_path(image_name, message, task_data):
                if image_name in message.model_fields_set and getattr(
                    message, image_name
                ):
                    image_path = getattr(message, image_name)
                    if image_path.startswith("http"):
                        image_path = await self.file_service.download_image(image_path)
                        task_data[image_name] = str(image_path)
                    elif is_base64_image(image_path):
                        image_path = save_base64_image(
                            image_path, str(self.file_service.input_image_dir)
                        )
                        task_data[image_name] = str(image_path)
                    else:
                        task_data[image_name] = image_path
                return task_data

            task_data = await update_image_path("first_image_path", message, task_data)
            task_data = await update_image_path("last_image_path", message, task_data)

            if "audio_path" in message.model_fields_set and message.audio_path:
                if message.audio_path.startswith("http"):
                    audio_path = await self.file_service.download_audio(
                        message.audio_path
                    )
                    task_data["audio_path"] = str(audio_path)
                elif is_base64_audio(message.audio_path):
                    audio_path = save_base64_audio(
                        message.audio_path, str(self.file_service.input_audio_dir)
                    )
                    task_data["audio_path"] = str(audio_path)
                else:
                    task_data["audio_path"] = message.audio_path

            actual_save_path = self.file_service.get_output_path(
                message.save_video_path
            )
            task_data["save_video_path"] = str(actual_save_path)
            task_data["video_path"] = message.save_video_path
            if not self.inference_service.submit_task(task_data):
                raise RuntimeError("Distributed inference service is not started")

            result = self.inference_service.wait_for_result_with_stop(
                message.task_id, stop_event, timeout=300
            )

            if result is None:
                if stop_event.is_set():
                    logger.info(f"Task {message.task_id} cancelled during processing")
                    return None
                raise RuntimeError("Task processing timeout")

            if result.get("status") == "success":
                return TaskResponse(
                    task_id=message.task_id,
                    task_status="completed",
                    save_video_path=message.save_video_path,  # Return original path
                )
            else:
                error_msg = result.get("error", "Inference failed")
                raise RuntimeError(error_msg)

        except Exception as e:
            logger.error(f"Task {message.task_id} processing failed: {str(e)}")
            raise
