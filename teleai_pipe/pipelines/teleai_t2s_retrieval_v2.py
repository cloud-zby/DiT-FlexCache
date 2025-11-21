import os
import re
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from abc import ABC
from collections import defaultdict

import torch
import numpy as np
from torch import Tensor
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from teleai_pipe.pipelines.base_pipeline import BasePipeline
from teleai_pipe.pipelines.teleai_t2s_v1 import TeleAIT2SPipeline
import json
import openai
import random

import httpx
from loguru import logger
from openai import (
    NOT_GIVEN,
    ContentFilterFinishReasonError,
    LengthFinishReasonError,
    NotGiven,
    OpenAI,
)
from PIL.ImageFile import ImageFile
from pydantic import BaseModel, Field
from typing import Optional, ClassVar

MODEL = "/nvfile-heatstorage/model_zoo/huggingface/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"


SCENE_MATCHING_EN_PROMPT = """Identify the paragraph index and matching score that aligns best with the video scene described by the given query.

Understand the Objective: From multiple paragraph scenarios describing video scenes, identify the paragraph with the most aligned scene based on the given query, and return the relevancy score through scene alignment criteria.

- Multiple paragraphs will be provided as index-content pairs, for example:
  - 0: paragraph1
  - 1: paragraph2
- Output requirement: Return a JSON object containing the index of the most aligned paragraph and the matching score.

# Steps

1. **Accept Input:** 
   - A series of paragraphs describing video scenes, each accompanied by an index.
   - A query string describing a scene.

2. **Scene Alignment Calculation:**
   - Consider scene alignment criteria such as location match, event similarity, and key character identification to compute a matching score for each paragraph. Scores range from 0 to 1.

3. **Select Optimal Result:**
   - Determine the paragraph index with the highest alignment score. If the highest score is below a reasonable threshold (e.g., 0.1), consider it as no match and return index -1 and score as 1.

4. **Form Output:**
   - Output format: `{"matched_index": xxx, "score": xxx}`

# Output Format

- JSON format, containing the following key-value pairs:
  - `"matched_index"`: Return the paragraph index that best aligns with the query scene. If there is no suitable match, the index is -1.
  - `"score"`: Score ranges from 0 to 1, with higher scores indicating better alignment. If there is no suitable match, the score is 1.

# Examples

Example Input: 

Paragraphs:
- 0: "The sun is shining, and we are going for a walk in the park."
- 1: "Last night, there was a heavy rain."

Query: "A sunny day at the park"

Example Output:
```json
{
  "matched_index": 0,
  "score": 0.75
}
```

# Notes

- If there is no obvious scene alignment between the query and the paragraph content, leading to a very low alignment score, use the default strategy to return index -1 and score as 1.
- Ensure that the scene alignment process considers contextual and event-specific factors to improve matching accuracy."""

SUBJECT_MATCHING_EN_PROMPT = """Identify the video subject index and matching score that best matches the query about the video's main topic, with less strict matching criteria.

Understand the Objective: From multiple scenarios describing video subjects, identify the video subject with the most aligned topic based on the given query, and return the relevancy score through alignment criteria with relaxed matching rules.

- Multiple video subjects will be provided as index-content pairs, for example:
  - 0: subject1
  - 1: subject2
- Output requirement: Return a JSON object containing the index of the most aligned video subject and the matching score.

# Steps

1. **Accept Input:** 
   - A series of video subjects, each accompanied by an index.
   - A query string describing the main topic.

2. **Subject Alignment Calculation:**
   - Consider more flexible alignment criteria such as partial topic similarity, broad keyword relevance, and general content overlap to compute a matching score for each video subject. Scores range from 0 to 1.

3. **Select Optimal Result:**
   - Determine the video subject index with the highest alignment score. If no suitable match is found, return index -1 and a score of 1 as default.

4. **Form Output:**
   - Output format: `{"matched_index": xxx, "score": xxx}`

# Output Format

- JSON format, containing the following key-value pairs:
  - `"matched_index"`: Return the subject index that best aligns with the query topic. If there is no suitable match, the index is -1.
  - `"score"`: Score ranges from 0 to 1, with higher scores indicating better alignment. If there is no suitable match, the score is 1.

# Examples

Example Input:

Video Subjects:
- 0: "A documentary about space exploration."
- 1: "A person surviving in the wild."

Query: "Survival techniques"

Example Output:
```json
{
  "matched_index": 1,
  "score": 0.5
}
```
(Note: Real examples should aim for flexible matches based on broad relevance.)

# Notes

- Ensure that the alignment process accommodates contextual flexibility and broader topic-specific factors to enhance matching applicability."""

SUBJECT_SIMILARITY_EN_PROMPT = """Assess the similarity of the subject descriptions in a given reference paragraph and a user query, focusing on the degree to which each describes the same video subject.

# Steps

1. **Subject Analysis**: Analyze both the reference paragraph and the user query to determine their shared subject(s) as related to video content.
2. **Description Comparison**: Assess how each text describes the subject(s) of the video, noting any differences or similarities in descriptions.
3. **Similarity Evaluation**: Evaluate the extent to which the descriptions align, identifying any discrepancies or congruities.
4. **Score Calculation**: Calculate a similarity score based on the degree of description alignment and differences.

# Output Format

Provide the result in the JSON format below with a score range from 0 to 1:

```json
{"similarity_score": xxxx}
```

# Examples

**Example 1:**

- **Reference Paragraph**: 
  [Reference text with a description of video subject goes here]
- **User Query**: 
  [User query text with a description of video subject goes here]

**Output:**

```json
{"similarity_score": 0.85}
```

# Notes

- Ensure the calculation considers how well the descriptions in both texts align regarding the video's subjects.
- The final score should reflect the overall similarity of subject descriptions between the texts.
- The scale must always be between 0 (no similarity) and 1 (perfect similarity)."""

SCRIPT_SPLITER_EN_PROMPT = """Break down the user's description based on the provided "input" and "acceptable action type" into four fields: Action Type, Scene, Subject, and Is Success. Ensure each field accurately reflects the type of action, the scene where it occurs, the characteristics of the person performing the action, and the overall success status based on the provided constraints. If "scene" or "subject" is not specifically outlined in the description, assign a specific default flag for them.

# Steps

1. **Understand the Input Description**: Carefully read the description provided by the user.
2. **Determine Scene**: Extract the environment or location where the action takes place from the description or assign a default flag if not specified.
3. **Describe Subject**: Identify the character involved in the description, include their physical characteristics, or assign a default flag if not specified.
4. **Identify Action Type**: Determine the main type of action included in the input description. This must be selected from a pre-defined list of acceptable actions.
5. **Assess Success**: Set `is_success` to true only if the `action_type` can be matched to the predefined list of acceptable actions. If no match is found, set `is_success` to false and assign an appropriate `action_type`.

# Output Format

Produce a JSON object containing the following four fields:
- `scene`: Details of the environment where the action takes place. If not provided, assign a default flag "unspecified_scene".
- `subject`: The appearance or identity of the individual performing the action. If not provided, assign a default flag "unspecified_subject".
- `action_type`: The described type of action, selected from a pre-defined list.
- `is_success`: A boolean indicating if the action was successfully identified from the list.

# Examples

### Example

- **Acceptable Action Type**:
swimming, dancing, running, diving_water
- **Input**: 
"On the beach at sunset, a young woman with a red shirt is swimming."

Output:
```json
{
  "scene": "a beach at sunset",
  "subject": "a young woman with a red shirt",
  "action_type": "swimming",
  "is_success": true
}
```

- **Input**: 
"The man is running."

Output:
```json
{
  "scene": "unspecified_scene",
  "subject": "the man",
  "action_type": "running",
  "is_success": true
}
```

- **Input**: 
"It was a quiet morning."

Output:
```json
{
  "scene": "a quiet morning",
  "subject": "unspecified_subject",
  "action_type": "unspecified",
  "is_success": false
}
```

# Notes

- Ensure that the information for each field is accurately extracted from the input.
- The `action_type` must be selected from the given list of specific and common action vocabulary.
- Scene information should include as much detail as possible, including time and location.
- The subject's characteristics should be as complete as possible to allow unique identification.
- Default flags should be used for "scene" and "subject" if they are not explicitly mentioned.
"""


class SceneMatchTaskOutput(BaseModel):
    matched_index: int
    score: float = Field(..., ge=0, le=1)


class SubjectMatchTaskOutput(SceneMatchTaskOutput):
    pass


class SubjectSimilarityOutput(BaseModel):
    similarity_score: float = Field(..., ge=0, le=1)


class ScriptSplitOutput(BaseModel):
    scene: str
    subject: str
    action_type: str
    is_success: bool


class MatchResult(SceneMatchTaskOutput):
    matched_paragraph: str


class LLMTask(ABC):
    _data_model: ClassVar[BaseModel]

    def __init__(self, base_url: str, seed: int, temperature: float = 0.0) -> None:
        self._seed = seed

        self._num_words = 50
        self._temperature = temperature

        self.client = OpenAI(
            base_url=base_url,
            api_key="no-api-key-required",
            max_retries=0,
        )


class LLMSceneMatcher(LLMTask):
    _data_model = SceneMatchTaskOutput

    def _build_user_message(self, paragraphs: list[str], query: str) -> str:
        paragraphs_str = "\n".join(f'-{i}: "{p}"' for i, p in enumerate(paragraphs))

        return f'Paragraphs: \n{paragraphs_str}\nQuery: "{query}"'

    def _build_request(self, paragraphs: list[str], query: str, model: str) -> dict:
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SCENE_MATCHING_EN_PROMPT,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self._build_user_message(paragraphs, query),
                    },
                ],
            },
        ]

        return {
            "model": model,
            "messages": messages,
            "temperature": self._temperature,
            "seed": self._seed,
            "logprobs": True,
            "response_format": self._data_model,
        }

    def __call__(self, paragraphs: list[str], query: str, model: str) -> MatchResult:
        req = self._build_request(paragraphs, query, model)
        with self.client.beta.chat.completions.stream(
            **req,
            timeout=60,
            max_tokens=50,
            extra_body={"repetition_penalty": 1.2},
        ) as st:
            res = st.get_final_completion()
            try:
                match_output = res.choices[0].message.parsed
            except Exception as e:
                return None

            if match_output is None:
                return None
            if match_output.matched_index == -1:
                return None
            if match_output.score < 0.1:
                return None

            return MatchResult(
                matched_index=match_output.matched_index,
                score=match_output.score,
                matched_paragraph=paragraphs[match_output.matched_index],
            )


class LLMSubjectMatcher(LLMTask):
    _data_model = SubjectMatchTaskOutput

    def _build_user_message(self, paragraphs: list[str], query: str) -> str:
        paragraphs_str = "\n".join(f'-{i}: "{p}"' for i, p in enumerate(paragraphs))

        return f'Paragraphs: \n{paragraphs_str}\nQuery: "{query}"'

    def _build_request(self, paragraphs: list[str], query: str, model: str) -> dict:
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SUBJECT_MATCHING_EN_PROMPT,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self._build_user_message(paragraphs, query),
                    },
                ],
            },
        ]

        return {
            "model": model,
            "messages": messages,
            "temperature": self._temperature,
            "seed": self._seed,
            "logprobs": True,
            "response_format": self._data_model,
        }

    def __call__(self, paragraphs: list[str], query: str, model: str) -> MatchResult:
        req = self._build_request(paragraphs, query, model)
        with self.client.beta.chat.completions.stream(
            **req,
            timeout=60,
            max_tokens=50,
            extra_body={"repetition_penalty": 1.2},
        ) as st:
            res = st.get_final_completion()
            try:
                match_output = res.choices[0].message.parsed
            except Exception as e:
                return None

            if match_output is None:
                return None
            if match_output.matched_index == -1:
                return None
            if match_output.score < 0.1:
                return None

            return match_output


class LLMSubjectSimilarityMatcher(LLMTask):
    _data_model = SubjectSimilarityOutput

    def _build_user_message(self, reference: str, query: str) -> str:
        return f'- **Reference Paragraph**: \n{reference}\n- **User Query**: "{query}"'

    def _build_request(self, reference: str, query: str, model: str) -> dict:
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SUBJECT_SIMILARITY_EN_PROMPT,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self._build_user_message(reference, query),
                    },
                ],
            },
        ]

        return {
            "model": model,
            "messages": messages,
            "temperature": self._temperature,
            "seed": self._seed,
            "logprobs": True,
            "response_format": self._data_model,
        }

    def __call__(self, reference: str, query: str, model: str) -> bool:
        req = self._build_request(reference, query, model)
        with self.client.beta.chat.completions.stream(
            **req,
            timeout=60,
            max_tokens=50,
            extra_body={"repetition_penalty": 1.2},
        ) as st:
            res = st.get_final_completion()
            try:
                output = res.choices[0].message.parsed
            except Exception as e:
                return False

            if output is None:
                return False
            if output.similarity_score < 0.5:
                return False
            else:
                return True


class LLMScriptSplitter(LLMTask):
    _data_model = ScriptSplitOutput

    def _build_user_message(
        self, acceptable_action_types: list[str], query: str
    ) -> str:
        types = ",".join(acceptable_action_types)
        return f'- **Acceptable Action Type**: "{types}"\n- **Input**:  \n{query}'

    def _build_request(
        self, acceptable_action_types: list[str], query: str, model: str
    ) -> dict:
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SCRIPT_SPLITER_EN_PROMPT,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self._build_user_message(
                            acceptable_action_types, query
                        ),
                    },
                ],
            },
        ]

        return {
            "model": model,
            "messages": messages,
            "temperature": self._temperature,
            "seed": self._seed,
            "logprobs": True,
            "response_format": self._data_model,
        }

    def __call__(
        self, acceptable_action_types: list[str], query: str, model: str
    ) -> Optional[ScriptSplitOutput]:
        req = self._build_request(acceptable_action_types, query, model)
        with self.client.beta.chat.completions.stream(
            **req,
            timeout=60,
            extra_body={"repetition_penalty": 1.2},
        ) as st:
            res = st.get_final_completion()
            try:
                output = res.choices[0].message.parsed
            except Exception as e:
                return None

            if output is None:
                return None

            return output


class MotionDatabase:
    """Database for storing and retrieving motion data based on descriptions."""

    def __init__(
        self,
        base_dir: str,
        cache_file: str = "motion_database_cache.pkl",
        llm_base_url: str = "http://10.127.16.103:1234/v1",
        model_name_or_path: str = "/nvfile-heatstorage/model_zoo/huggingface/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
    ):
        self.base_dir = Path(base_dir)
        self.cache_file = cache_file

        # Initialize folders
        self.poses_dir = self.base_dir / "poses"
        self.first_dir = self.base_dir / "first"
        self.segment_dir = self.base_dir / "segment"
        self.action_type_dir = self.base_dir / "action_type"
        self.subject_dir = self.base_dir / "subject"
        self.scene_dir = self.base_dir / "scene"

        # Initialize Qwen client
        self.llm_paragraph_mather = LLMSceneMatcher(
            llm_base_url, seed=None, temperature=1.0
        )
        self.llm_subject_mather = LLMSubjectMatcher(
            llm_base_url, seed=None, temperature=1.8
        )
        self.llm_paragraph_similarity_matcher = LLMSubjectSimilarityMatcher(
            llm_base_url, seed=None, temperature=1.0
        )
        self.llm_script_splitter = LLMScriptSplitter(
            llm_base_url, seed=None, temperature=1.0
        )
        self.model_name_or_path = model_name_or_path

        # Data structures for our database
        self.motion_data_list = []  # Main data list
        self.action_type_index = {}  # Action type -> list of indices
        self.scene_index = {}  # Action type -> scene -> list of indices
        self.subject_index = {}  # "Action_Scene" -> subject -> list of indices

        # Try to load from cache first
        if os.path.exists(cache_file):
            self.load_from_cache()
        else:
            self.build_database()

    def load_from_cache(self):
        """Load the database from cache file."""
        try:
            with open(self.cache_file, "rb") as f:
                cached_data = pickle.load(f)
                self.motion_data_list = cached_data.get("motion_data_list", [])
                self.action_type_index = cached_data.get("action_type_index", {})
                self.scene_index = cached_data.get("scene_index", {})
                self.subject_index = cached_data.get("subject_index", {})
            print(
                f"Loaded database from cache with {len(self.motion_data_list)} entries"
            )
            return True
        except Exception as e:
            print(f"Error loading cache: {e}")
            self.build_database()
            return False

    def save_to_cache(self):
        """Save the database to cache file."""
        try:
            cache_data = {
                "motion_data_list": self.motion_data_list,
                "action_type_index": self.action_type_index,
                "scene_index": self.scene_index,
                "subject_index": self.subject_index,
            }
            with open(self.cache_file, "wb") as f:
                pickle.dump(cache_data, f)
            print(f"Saved database to cache with {len(self.motion_data_list)} entries")
            return True
        except Exception as e:
            print(f"Error saving cache: {e}")
            return False

    def build_database(self):
        """Build the database by reading all folders."""
        print("Building motion database...")

        # Get list of all motion files
        motion_files = list(self.poses_dir.glob("*.pkl"))

        for motion_file in motion_files:
            filename = motion_file.stem

            # Check if all corresponding files exist
            first_frame = list(self.first_dir.glob(f"{filename}.*"))
            segment_mask = list(self.segment_dir.glob(f"{filename}.*"))
            action_type_file = self.action_type_dir / f"{filename}.txt"
            subject_file = self.subject_dir / f"{filename}.txt"
            scene_file = self.scene_dir / f"{filename}.txt"

            if (
                first_frame
                and segment_mask
                and action_type_file.exists()
                and subject_file.exists()
                and scene_file.exists()
            ):
                # Read the descriptions
                with open(action_type_file, "r", encoding="utf-8") as f:
                    action_type = f.read().strip()

                with open(subject_file, "r", encoding="utf-8") as f:
                    subject = f.read().strip()

                with open(scene_file, "r", encoding="utf-8") as f:
                    scene = f.read().strip()

                # Create data entry
                data_index = len(self.motion_data_list)
                self.motion_data_list.append(
                    {
                        "id": data_index,
                        "action_type": action_type,
                        "scene": scene,
                        "subject": subject,
                        "motion_file": str(motion_file),
                        "image_file": str(first_frame[0]),
                        "segment_file": str(segment_mask[0]),
                    }
                )

                # Update action type index
                if action_type not in self.action_type_index:
                    self.action_type_index[action_type] = []
                self.action_type_index[action_type].append(data_index)

                # Update scene index
                if action_type not in self.scene_index:
                    self.scene_index[action_type] = {}
                if scene not in self.scene_index[action_type]:
                    self.scene_index[action_type][scene] = []
                self.scene_index[action_type][scene].append(data_index)

                # Update subject index
                key = f"{action_type}_{scene}"
                if key not in self.subject_index:
                    self.subject_index[key] = {}
                if subject not in self.subject_index[key]:
                    self.subject_index[key][subject] = []
                self.subject_index[key][subject].append(data_index)

        print(f"Database built with {len(self.motion_data_list)} entries")
        self.save_to_cache()

    def get_class_database(self, action_type: str) -> dict:
        """
        Get all scenes and subjects for a given action type.

        Args:
            action_type: The action type to search for

        Returns:
            dict: {"scene": list of scenes, "subject": list of subjects}
        """
        # Find matching action type
        matched_action = action_type

        # Get all scenes for this action type
        scenes = list(self.scene_index.get(matched_action, {}).keys())

        # Get all unique subjects across all scenes
        subjects = []
        for scene in scenes:
            key = f"{matched_action}_{scene}"
            subject = list(self.subject_index.get(key, {}).keys())[0]
            subjects.append(subject)

        return {"scene": scenes, "subject": list(subjects)}

    def _load_data(self, data_index) -> dict:
        """
        Load motion data, image file, and segment file for a given data index.

        Args:
            data_index: Index in the motion_data_list or None/-1

        Returns:
            dict: {"poses": motion_data, "image": image_path, "segment": segment_path}
        """
        if data_index is None or data_index == -1:
            return {"poses": None, "image": None, "segment": None}

        # Get data from the list
        data = self.motion_data_list[data_index]

        # Load motion data
        try:
            with open(data["motion_file"], "rb") as f:
                motion_data = pickle.load(f)
        except Exception as e:
            print(f"Error loading motion file: {e}")
            motion_data = None

        return {
            "poses": motion_data,
            "image": data["image_file"],
            "segment": data["segment_file"],
        }

    def get_available_action_types(self) -> list:
        """
        获取数据库中所有可用的动作类型列表。

        Returns:
            list: 所有可用的动作类型列表
        """
        # 直接返回动作类型索引的键列表
        action_types = list(self.action_type_index.keys())

        # 可以选择按字母顺序排序以提供更一致的输出
        action_types.sort()

        print(f"Found {len(action_types)} available action types")
        return action_types

    # Please achieve it by qwen2.5
    def retrieval(self, user_prompt: str) -> dict:
        """

        Process a user's English prompt and perform hierarchical matching.

        Args:
            user_prompt: The user's English prompt

        Returns:
            dict: Results containing code and matched data
        """
        # Step 1: Parse user prompt and Match action type
        acceptable_action_types = self.get_available_action_types()
        parsed = self.llm_script_splitter(
            acceptable_action_types, user_prompt, self.model_name_or_path
        )

        if parsed is None or parsed.is_success is False:
            # Action type matching failed - code 0
            print(f"Faild to match: {parsed}")
            return {"code": 0, "motion": None, "segment": None, "first_image": None}

        print("Parsed descriptions:")
        print(f"  Action: {parsed.action_type}")
        print(f"  Scene: {parsed.scene}")
        print(f"  Subject: {parsed.subject}")

        dicts_scene_subject = self.get_class_database(parsed.action_type)

        # Step 2: Match scene
        sene_database = dicts_scene_subject["scene"]

        shuffled_indices = list(range(len(sene_database)))
        random.shuffle(shuffled_indices)

        sene_database = [sene_database[i] for i in shuffled_indices]
        subject_database = [dicts_scene_subject["subject"][i] for i in shuffled_indices]

        has_subject = False
        if parsed.scene == "unspecified_scene":
            # If the scene is unspecified, return a random scene
            idx = None
            if parsed.subject != "unspecified_subject":
                matched_subject = self.llm_subject_mather(
                    subject_database, parsed.subject, self.model_name_or_path
                )
                if matched_subject is not None:
                    has_subject = True
                    idx = matched_subject.matched_index
            if idx is None:
                for i, scene in enumerate(sene_database):
                    key = f"{parsed.action_type}_{scene}"
                    subject = list(self.subject_index.get(key, {}).keys())[0]
                    data_index = self.subject_index[key][subject][0]
                    data = self._load_data(data_index)
                    if not data["segment"].endswith(".npz"):
                        idx = i
                        break
                if idx is None:
                    idx = random.randint(0, len(sene_database) - 1)
            matched_scene = MatchResult(
                matched_index=idx, score=1.0, matched_paragraph=sene_database[idx]
            )
        else:
            matched_scene = self.llm_paragraph_mather(
                sene_database, parsed.scene, self.model_name_or_path
            )
        # 需要实现场景检索

        if matched_scene is None:
            # Scene matching failed - code 3
            # Return random pose from the action type category
            if (
                parsed.action_type in self.action_type_index
                and self.action_type_index[parsed.action_type]
            ):
                # Get a random index from the action type
                random_idx = random.choice(self.action_type_index[parsed.action_type])
                data = self._load_data(random_idx)
                return {
                    "code": 3,
                    "motion": data["poses"]["poses"],
                    "segment": None,
                    "first_image": data["image"],
                }

        print(f"Matched scene: {matched_scene}")

        # Step 3: Match subject
        if parsed.subject == "unspecified_subject":
            is_matched_subject = True
        else:
            if has_subject is False:
                is_matched_subject = self.llm_paragraph_similarity_matcher(
                    subject_database[matched_scene.matched_index],
                    parsed.subject,
                    self.model_name_or_path,
                )
            else:
                is_matched_subject = True

        key = f"{parsed.action_type}_{matched_scene.matched_paragraph}"

        if is_matched_subject is False:
            # Subject matching failed - code 2
            # Still return the scene match data
            if key in self.subject_index and self.subject_index[key]:
                # Get the first subject and its first index
                first_subject = next(iter(self.subject_index[key].keys()))
                data_index = self.subject_index[key][first_subject][0]
                data = self._load_data(data_index)
                if not data["segment"].endswith(".npz"):
                    return {
                        "code": 2,
                        "motion": data["poses"]["poses"],
                        "segment": data["segment"],
                        "first_image": data["image"],
                    }
                return {
                    "code": 3,
                    "motion": data["poses"]["poses"],
                    "segment": data["segment"],
                    "first_image": data["image"],
                }

        matched_subject = subject_database[matched_scene.matched_index]
        print(f"Matched subject: {matched_subject}")

        # Step 5: All matches successful - code 1
        # Get the first matching data index

        data_index = self.subject_index[key][matched_subject][0]
        data = self._load_data(data_index)

        return {
            "code": 1,
            "motion": data["poses"]["poses"],
            "segment": None,
            "first_image": data["image"],
        }


def get_image_size(img_path: str) -> Tuple[int, int]:
    import cv2

    img = cv2.imread(img_path)
    return img.shape[:2][::-1]


def norm_pose(pose: Tensor, img_size: Tuple[int, int] = (1920, 1080)):
    return pose / torch.tensor(img_size, dtype=torch.float32).to(device=pose.device)


def unnorm_pose(pose: Tensor, img_size: Tuple[int, int] = (1920, 1080)):
    return pose * torch.tensor(img_size, dtype=torch.float32).to(device=pose.device)


class TeleAIT2SRetrivalPipeline(BasePipeline):
    def __init__(
        self,
        t2s_vae_path: str,
        t2s_tf_path: str,
        database_dir: str,
        database_cache_file: str,
        llm_base_url: str = "http://10.127.16.103:1234/v1",
        model_name_or_path: str = "/nvfile-heatstorage/model_zoo/huggingface/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
    ) -> None:
        super().__init__()
        # 设置路径
        # 创建匹配器实例
        # matcher = MotionTextMatcher(text_dir, motion_dir, model_dir, threshold=0.7)
        self.motion_database = MotionDatabase(
            database_dir, database_cache_file, llm_base_url, model_name_or_path
        )

        self.t2s_pipeline = TeleAIT2SPipeline.from_pretrained(
            t2s_vae_path, t2s_tf_path, device="cuda", torch_dtype=torch.float32
        )

    def _interpolate(self, motion: Tensor, n: int) -> Tensor:
        N, _, _ = motion.shape
        inter_frames = []
        for i in range(N - 1):
            inter_frames.append(motion[i])
            cur_frame = motion[i]
            next_frame = motion[i + 1]
            for j in range(1, n + 1):
                alpha = j / (n + 1)
                inter_frame = cur_frame * (1 - alpha) + alpha * next_frame
                inter_frames.append(inter_frame)
        inter_frames.append(motion[-1])
        inter_motion = np.array(inter_frames)
        inter_motion = torch.tensor(inter_motion)
        return inter_motion

    def _shift_motion_by_first_pose(
        self,
        motion: torch.Tensor,
        first_pose: torch.Tensor,
        pelvis_joint_idx: int = 0,
    ) -> torch.Tensor:
        """
        使用 bounding box 来对齐和缩放生成的动作序列，使其首帧与目标 pose 对齐，避免相机抖动。

        参数：
            generated_motion: Tensor，shape [T, J, D]，归一化后的动作序列
            first_pose_gt: Tensor，shape [J, D]，目标首帧 pose（未归一化）
            pelvis_joint_idx: int，骨盆索引
        返回：
            adjusted_motion: Tensor，shape [T, J, D]，后处理后的动作
        """
        # Step 1: 目标首帧的最大边长（用于对比）
        target_bbox_size = first_pose.max(dim=0).values - first_pose.min(dim=0).values
        target_size = target_bbox_size.max()  # scalar

        # Step 2: 所有帧的 bbox 最大边长
        min_coords = motion.amin(dim=1)  # [T, D]
        max_coords = motion.amax(dim=1)  # [T, D]
        bbox_sizes = max_coords - min_coords  # [T, D]
        frame_scales = bbox_sizes.max(dim=1).values  # [T]

        # Step 3: 计算平均缩放比例
        scale_ratios = target_size / frame_scales  # [T]
        scale_ratio = (scale_ratios.sum() - scale_ratios.max() - scale_ratios.min()) / (
            scale_ratios.shape[0] - 2
        )  # scalar

        # Step 4: 缩放
        scaled_motion = motion * scale_ratio

        # Step 5: 对齐 pelvis
        pelvis_gen = scaled_motion[0, pelvis_joint_idx]  # [D]
        pelvis_gt = first_pose[pelvis_joint_idx]  # [D]
        pelvis_offset = pelvis_gt - pelvis_gen  # [D]

        # Step 6: 平移所有帧
        adjusted_motion = scaled_motion + pelvis_offset  # broadcasting

        return adjusted_motion

    def _extract_main_pose(self, motions: list) -> Tensor:
        def find_largest_bbox_pose_torch(pose_tensor):
            """
            输入: pose_tensor 是一个形状为 (n, 24, 2) 的 torch.Tensor
            输出: 最大bbox面积对应的索引 和 对应的bbox面积
            """
            # 分别取出 x 和 y 坐标
            x = pose_tensor[:, :, 0]  # (n, 24)
            y = pose_tensor[:, :, 1]  # (n, 24)

            # 获取每个 pose 的 min 和 max
            x_min = x.min(dim=1).values
            x_max = x.max(dim=1).values
            y_min = y.min(dim=1).values
            y_max = y.max(dim=1).values

            # 计算宽度和高度
            widths = x_max - x_min
            heights = y_max - y_min

            # 计算 bbox 面积
            areas = widths * heights

            # 找出最大面积的索引
            max_area, max_index = areas.max(dim=0)

            return max_index.item(), max_area.item()

        first_person_poses = []
        for pose in motions:
            if pose.shape[0] == 0:
                continue
                # first_person_poses.append(pose)
            pose_tensor = torch.tensor(pose.tolist())
            index, _ = find_largest_bbox_pose_torch(pose_tensor)
            first_person_poses.append(pose_tensor[index].tolist())
        motion = torch.tensor(first_person_poses)
        return motion

    def _is_pose_normed(self, motion: Tensor) -> bool:
        return (motion.max() <= 1).item()

    def postprocess(
        self, control: Optional[Tensor], gen_data: dict, is_retrieval: bool
    ) -> Tensor:
        if is_retrieval and gen_data["motion"]:
            self._retrieval_postprocess(control, gen_data)
        # elif control is not None:
        #     gen_data["motion"] = self._interpolate(gen_data["motion"], 1)

    def _retrieval_postprocess(
        self, control: Optional[Tensor], retrieval_data: dict
    ) -> Tensor:
        gen_motion = retrieval_data["motion"]
        retrieval_image_size = get_image_size(retrieval_data["first_image"])
        retrieval_data["original_motion"] = [poses.tolist() for poses in gen_motion]
        gen_motion = self._extract_main_pose(gen_motion)
        if not self._is_pose_normed(gen_motion):
            if control is None:
                retrieval_data["motion"] = gen_motion
                retrieval_data["retrieved_motion"] = gen_motion
                return
            normd_gen_motion = norm_pose(gen_motion, retrieval_image_size)
        else:
            normd_gen_motion = gen_motion

        shifted_motion = self._shift_motion_by_first_pose(normd_gen_motion, control)
        if retrieval_data["code"] in []:
            gen_motion = self.t2s_pipeline.edit(control, shifted_motion)
        else:
            gen_motion = torch.concat([control.unsqueeze(0), shifted_motion])

        gen_motion = unnorm_pose(gen_motion, retrieval_image_size)
        retrieval_data["motion"] = gen_motion
        retrieval_data["retrieved_motion"] = unnorm_pose(
            torch.concat([control.unsqueeze(0), normd_gen_motion]), retrieval_image_size
        )

    def __call__(
        self, prompt, cn_translation, control, num_frames, infer_opt
    ) -> Tuple[bool, dict]:
        is_retrieval = False
        # cn_data = self.motion_database.retrieval(cn_translation)
        data = self.motion_database.retrieval(prompt)
        # data = cn_data if cn_data is not None else en_data
        if data is None:
            gen_motion = self.t2s_pipeline(prompt, control, num_frames, infer_opt)[0]
            data = {"motion": gen_motion}
        elif data["code"] != 0:
            is_retrieval = True

        if control is not None:
            self.postprocess(control.cpu().detach(), data, is_retrieval=is_retrieval)
        else:
            self.postprocess(None, data, is_retrieval=is_retrieval)
        print(data)
        return is_retrieval, data


if __name__ == "__main__":
    # Initialize the database
    db = MotionDatabase(
        base_dir="/nvfile-heatstorage/hzh/motion/jchen/0430_motion_database",
        llm_base_url="http://10.127.16.1:1234/v1",
        model_name_or_path="/nvfile-heatstorage/model_zoo/huggingface/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
    )

    print(db.motion_data_list)

    # Process a user prompt
    result = db.retrieval("A woman is doing a back dive.")
    print(f"Code: {result['code']}")
    print(result["first_image"])

    result = db.retrieval("A woman is doing a back dive.")
    print(f"Code: {result['code']}")
    print(result["first_image"])

    result = db.retrieval("A woman is doing a back dive.")
    print(f"Code: {result['code']}")
    print(result["first_image"])

    # # Check the result code and use data accordingly
    # if result["code"] == 1:
    #     # All matches successful - use the data
    #     # print(f"Found matching motion: {result['pose']}")
    #     print(f"Image file: {result['first_image']}")
    #     print(f"Segment file: {result['segment']}")
    # elif result["code"] == 0:
    #     print("No matching action type found")
    # elif result["code"] == 3:
    #     print("Action type matched but no matching scene found")
    # elif result["code"] == 2:
    #     print("Action type and scene matched but no matching subject found")
