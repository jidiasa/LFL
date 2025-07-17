from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torchvision.transforms import ToPILImage

from abc import ABC, abstractmethod
from typing import Callable, Union, List
from agents import ChatGPT4Agent

class KeyframeGenerator(ABC):

    @abstractmethod
    def __call__(
        self,
        id,
        start_kf_img: "torch.Tensor",
        *,
        modify_prompt: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        pass
        
class _CallableWrapper(KeyframeGenerator):

    def __init__(self, fn: Callable[..., Dict[str, Any]]):
        self.fn = fn

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # noqa: D401
        return self.fn(*args, **kwargs)
    
class KeyframeModifier(ABC):

    @abstractmethod
    def __call__(
        self,
        scene_idx: int,
        kf_id: int,
        start_kf_img: "torch.Tensor | PIL.Image.Image",
        *,
        original_image: torch.Tensor | None,
        mask: torch.Tensor | None,
        modify_prompt: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any] | None:  # may return updated kf_gen_dict
        raise NotImplementedError
    
class _CallableMod(KeyframeModifier):
    def __init__(self, fn: Callable[..., Dict[str, Any] | None]):
        self.fn = fn

    def __call__(self, *a, **kw):
        return self.fn(*a, **kw)
    
class RecursiveSceneGenerator:

    def __init__(
        self,
        config: Dict[str, Any],
        keyframe_generator: Union[KeyframeGenerator, Callable[..., Dict[str, Any]]],
        keyframe_modifier: KeyframeModifier | Callable[..., Dict[str, Any] | None] | None = None,
        *,
        start_keyframe: Any = None,
        adaptive_negative_prompt: str = "",
        rollback_steps: int = 1,
        max_feedback_rounds: int = 25,
    ) -> None:  
        # Strategy injection --------------------------------------------------
        if not isinstance(keyframe_generator, KeyframeGenerator):
            keyframe_generator = _CallableWrapper(keyframe_generator)
        if not isinstance(keyframe_modifier, KeyframeModifier):
            keyframe_modifier = _CallableMod(keyframe_modifier)    
        self.keyframe_gen = keyframe_generator
        self.kf_mod_strategy = keyframe_modifier

        # General config / state ---------------------------------------------
        self.adaptive_negative_prompt = adaptive_negative_prompt
        self.rollback_steps = rollback_steps
        self.max_feedback_rounds = max_feedback_rounds

        # Runtime accumulators -----------------------------------------------
        self.dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        self.save_dir = Path(config["runs_dir"]) / self.dt_string

        self.kf_gens: List[Dict[str, Any]] = []
        self.all_kf: List[Any] = []          # PIL images for quick access
        self.all_render: List[Any] = []      # raw tensors for modify()
        self.all_keyframes_data: List[Dict[str, Any]] = []
        self.mis: int = -1                   # last modified index

        self.scene_prompts: List[str] = []
        self.agent: ChatGPT4Agent = ChatGPT4Agent(
            root_path=self.save_dir,
            control=config.get("control", False),
        )

    def generate_kf_gen(self, id: int, start_kf, modify_prompt=""):
        return self.keyframe_gen(id, start_kf, modify_prompt=modify_prompt)

    def detect_scene_issue(self, img):
        has_issue, issue_description = self.agent.detect_iss(ToPILImage()(img))
        if has_issue:
            print(f"Detected issues: {issue_description}")
        else:
            print("No issues detected.")
        return has_issue, issue_description

    def detect_same_issue(self, img, iss_des):
        has_issue, window = self.agent.check_image_issue_with_window(
            ToPILImage()(img), iss_des
        )
        if has_issue:
            mask = torch.zeros((512, 512), dtype=torch.float32)
            print(window)
            mask[window["y_min"] : window["y_max"], window["x_min"] : window["x_max"]] = 1.0
            new_prompt = window["prompt"]
            return has_issue, mask, new_prompt
        mask = torch.zeros((512, 512), dtype=torch.float32)
        new_prompt = " "
        print("No issues detected.")
        return has_issue, mask, new_prompt

    def modify(
        self,
        id,
        start_kf,
        *,
        original_image=None,
        mask=None,
        modify_prompt="",
    ):
        pass 

    def _generate_recursive(self, id: int, feedback_round: int, issues: str = ""):
        flag = False
        if id > self.config["num_keyframes"] - 1:
            print("Scene generation completed.")
            return self.all_keyframes_data, self.all_kf, self.save_dir

        if feedback_round > self.max_feedback_rounds:
            print(f"Max feedback rounds reached at frame {id}. Skipping further repair.")
            flag = True

        start_kf = self.start_keyframe if id == 0 else ToPILImage()(self.kf_gens[id - 1]["kf2_image"][0])
        kf_gen = self.generate_kf_gen(id, start_kf)
        print(f"Generated frame: frame {id}")

        if id < len(self.kf_gens):
            self.kf_gens[id] = kf_gen
        else:
            self.kf_gens.append(kf_gen)

        iss, res = self.detect_scene_issue(kf_gen["kf2_image"][0])
        if iss and not flag:
            print(f"Issues at frame {id}: {iss}")
            rollback_start = max(0, id - self.rollback_steps)
            print(
                f"Rolling back to frame {rollback_start}"
            )

            for i in range(rollback_start, id + 1):
                if i <= self.mis:
                    if rollback_start != id:
                        rollback_start = i + 1
                    continue
                issues_i, mask, new_prompt = self.detect_same_issue(
                    self.kf_gens[i]["kf2_image"][0], res
                )
                if issues_i:
                    print(f"modifying frame {i}, Rewriting at frame {i+1}")
                    self.mis = i
                    start_kf_mod = (
                        self.start_keyframe if i == 0 else ToPILImage()(self.kf_gens[i - 1]["kf2_image"][0])
                    )
                    os.makedirs(self.save_dir, exist_ok=True)
                    self.all_kf[i].save(self.save_dir / f"modify_{i}.png")
                    self.modify(
                        i,
                        start_kf=start_kf_mod,
                        original_image=self.all_render[i],
                        mask=mask,
                        modify_prompt=new_prompt,
                    )
                    self.all_kf[i].save(self.save_dir / f"modified_{i}.png")
                    break
                if rollback_start != id:
                    rollback_start = i + 1

            self.kf_gens = self.kf_gens[: rollback_start + 1]
            self.all_kf = self.all_kf[: rollback_start + 1]
            self.all_render = self.all_render[: rollback_start + 1]
            self.all_keyframes_data = self.all_keyframes_data[: rollback_start + 1]

            return self._generate_recursive(
                t=rollback_start + 1,
                feedback_round=feedback_round + 1,
                issues=iss,
            )
        return self._generate_recursive(id + 1, feedback_round, issues=issues)

    def generate(self):
        self._generate_recursive(t=0, feedback_round=0)
        os.makedirs(self.save_dir, exist_ok=True)
        for idx, img in enumerate(self.all_kf):
            img.save(self.save_dir / f"keyframe_{idx}.png")
        return self.all_keyframes_data, self.all_kf, self.save_dir
