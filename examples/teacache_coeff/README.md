## Teacache Coefficient Data Statistics

### Data Statistics Module

```python
from teleai_pipe.plugin.teacache.teacache import TeaCacheCalibrator
```
## Usage
When inference, add:
```python
    pipe.dit.cond_tea_cache = TeaCacheCalibrator(
        num_inference_step, 
        result_output_file,
        model_type,
        prompt_id
    )
    pipe.dit.uncond_tea_cache = TeaCacheCalibrator(
        num_inference_step, 
        result_output_file,
        model_type,
        prompt_id
    )
```
## Returned Information
```json
{
  "name": "model_type",
  "items": [
    {
      "prompt_id": 0,
      "calibrate_data": {
        "input_diffs": [],
        "res_output_diffs": [],
        "mod_input_diffs": [],
        "time_emb_diffs": []
      }
    },
    {
      "prompt_id": 1,
      "calibrate_data": {
        "input_diffs": [],
        "res_output_diffs": [],
        "mod_input_diffs": [],
        "time_emb_diffs": []
      }
    }
  ]
}
```
| Parameter               | Description                                  |
|-------------------------|----------------------------------------------|
| `mod_input_l1_diffs`    | timestep-embedding modulated noisy input    |
| `res_output_l1_diffs`   | the residual output                         |
| `input_l1_diffs`        | noisy input                                 |
| `time_emb_l1_diffs`     | the timestep embedding                      |

### Calculating Coefficient
```python
python get_coefficient.py
```
## Parameter Description
| Parameter       | Description                                       |
|----------------|---------------------------------------------------|
| `--result_dir`   | result result dir path                              |
| `--prefix`     | Result Output Path Prefix                         |
| `--order`      | Fitting Order                                     |


# Examples

```bash 
 python examples/teacache_coeff/teacache_pipe.py -t i2v -pf /nvfile-heatstorage/zgf/CODE/teleai_pipe/examples/teacache_coeff/wan_14b_image_to_video_h100.py -b /nvfile-heatstorage/zgf/CODE/teleai_pipe/examples/teacache_coeff/data.json -pt en -o work_dirs/teacache_result # 获得统计数据
 python examples/teacache_coeff/get_coefficient.py --result_dir work_dirs/teacache_pipe/teleai_video_v1_i2v_720p_49s_50_fps15_g6.0_eg1.0 # 根据统计信息获得参数
```