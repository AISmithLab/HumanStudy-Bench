---
name: Semi-Manual Study Generation Pipeline
overview: 半自动化研究生成pipeline，从论文PDF提取信息并生成兼容现有study001-004格式的输出。采用简洁的两阶段设计：Filter + Extraction，每个阶段输出可审核的markdown文件。
todos: []
---

# Semi-Manual Study Generation Pipeline

## 设计原则

1. **兼容性**: 生成的输出必须与现有study001-004格式完全兼容
2. **简洁性**: 遵循奥卡姆剃刀原则，最小化抽象层和代码复杂度
3. **封装**: 复用现有validation_pipeline的基础设施，避免重复代码

## 架构概述

```
PDF → Stage 1 (Filter) → Review.md → Stage 2 (Extract) → Review.md → JSON Files + Config Class
         ↑                                                                    ↓
         └────────────────────── Refine ────────────────────────────────────┘
```

## Pipeline结构

### 阶段1: Replicability Filter

**位置**: `generation_pipeline/filters/replicability_filter.py`

**功能**:
- 提取abstract
- 提取每个实验的简介（输入/参与者/输出）
- 判断可复现性（排除visual input、时间感知、模糊participant、无统计数据）

**输出**: `outputs/{paper_id}_stage1.md` + `.json`

**Review格式** (简化):
```markdown
# Stage 1: Replicability Filter

## Paper Info
- Title: [extracted]
- Authors: [extracted]
- Abstract: [extracted]

## Experiments

### Experiment 1: [name]
- Input: [what participants see]
- Participants: [description]
- Output: [what is measured]
- Replicable: YES/NO

#### Checklist:
- [ ] No visual input
- [ ] No time perception
- [ ] Participant profile constructible
- [ ] Has quantitative data

#### Comments:
[填写]

---

## Review Status
- Reviewed By: [填写]
- Action: [continue/refine/exclude]
```

### 阶段2: Study & Data Extraction

**位置**: `generation_pipeline/extractors/study_data_extractor.py`

**功能** (合并原阶段2+3):
- 提取所有study/problem
- 提取phenomenon
- 提取所有有统计数据的RQs
- 提取统计方法和结果（p值、效应量、描述性统计）
- 提取participant profile

**输出**: `outputs/{paper_id}_stage2.md` + `.json`

**Review格式** (简化):
```markdown
# Stage 2: Study & Data Extraction

## Study 1: [name]

### Phenomenon
[extracted]

### RQ1: [question]
- Statistical Method: [t-test/chi-square/etc.]
- Results: p=[value], statistic=[value], effect_size=[value]
- Descriptive: [means, SDs]

#### Checklist:
- [ ] Phenomenon correct
- [ ] Statistical data extracted
- [ ] Participant info extracted

#### Comments:
[填写]

---

## Review Status
- Reviewed By: [填写]
- Action: [continue/refine/back]
```

## 输出生成

### 1. JSON文件生成（兼容现有格式）

基于extraction结果，生成三个JSON文件（与study001-004格式一致）：

**metadata.json**:
```json
{
  "id": "study_XXX",
  "title": "[extracted]",
  "authors": ["[extracted]"],
  "year": [extracted],
  "domain": "[extracted]",
  "subdomain": "[extracted]",
  "keywords": ["[extracted]"],
  "difficulty": "[extracted]",
  "description": "[extracted]"
}
```

**specification.json**:
```json
{
  "study_id": "study_XXX",
  "title": "[extracted]",
  "participants": {
    "n": [extracted],
    "population": "[extracted]",
    "recruitment_source": "[extracted]",
    "demographics": {...}
  },
  "design": {
    "type": "[extracted]",
    "factors": [...]
  },
  "procedure": {...}
}
```

**ground_truth.json**:
```json
{
  "study_id": "study_XXX",
  "title": "[extracted]",
  "authors": ["[extracted]"],
  "year": [extracted],
  "validation_criteria": {
    "required_tests": [
      {
        "test_id": "P1",
        "test_type": "phenomenon_level",
        "method": {...}
      }
    ]
  },
  "original_results": {...}
}
```

### 2. StudyConfig类生成（统一接口）

**位置**: `generation_pipeline/generators/config_generator.py`

**功能**: 生成`BaseStudyConfig`子类，提供统一接口

**生成的代码模板** (简洁):
```python
@StudyConfigRegistry.register("study_XXX")
class StudyXXXConfig(BaseStudyConfig):
    def create_trials(self, n_trials=None):
        # 基于extracted design生成
        ...
    
    def aggregate_results(self, raw_results):
        # 基于extracted statistics聚合
        ...
```

**统一接口** (与现有study001-004一致):
```python
config = get_study_config(study_id, study_path, specification)
trials = config.create_trials()
instructions = config.get_instructions()
results = config.aggregate_results(raw_results)
```

## 实现细节

### 目录结构（最小化）

```
generation_pipeline/
├── pipeline.py              # 主orchestrator（简洁）
├── filters/
│   └── replicability_filter.py
├── extractors/
│   └── study_data_extractor.py
├── generators/
│   └── config_generator.py
├── utils/
│   ├── review_parser.py    # 解析markdown review
│   └── json_generator.py   # 生成兼容的JSON
└── outputs/
```

### 复用现有代码（最大化）

- **DocumentLoader**: 直接复用`validation_pipeline/utils/document_loader.py`
- **GeminiClient**: 直接复用`validation_pipeline/utils/gemini_client.py`
- **BaseStudyConfig**: 复用`src/core/study_config.py`
- **Review格式**: 参考`validation_pipeline/outputs/*_summary.md`但简化

### Pipeline Orchestrator（简洁设计）

**位置**: `generation_pipeline/pipeline.py`

```python
class GenerationPipeline:
    def __init__(self, llm_client):
        self.client = llm_client
        self.filter = ReplicabilityFilter(client)
        self.extractor = StudyDataExtractor(client)
        self.generator = ConfigGenerator()
    
    def run_stage1(self, pdf_path):
        """运行filter，输出review文件"""
        result = self.filter.process(pdf_path)
        self._save_review("stage1", result)
        return result
    
    def run_stage2(self, stage1_json):
        """运行extraction，输出review文件"""
        result = self.extractor.process(stage1_json)
        self._save_review("stage2", result)
        return result
    
    def generate_study(self, stage2_json, study_id):
        """生成JSON文件和Config类"""
        # 生成metadata.json, specification.json, ground_truth.json
        # 生成{study_id}_config.py
        ...
```

### Review Parser（简洁）

**位置**: `generation_pipeline/utils/review_parser.py`

```python
def parse_review(md_file: Path) -> Dict:
    """解析markdown review文件，提取checklist和comments"""
    # 简单解析，提取：
    # - Checklist状态
    # - Comments内容
    # - Review Status的action
    ...
```

### JSON Generator（兼容性）

**位置**: `generation_pipeline/utils/json_generator.py`

**功能**: 基于extraction结果生成与study001-004格式完全一致的JSON文件

**关键**: 确保字段名称、结构、嵌套层级与现有格式完全匹配

## 兼容性保证

### 1. JSON格式兼容

- **metadata.json**: 字段完全匹配（id, title, authors, year, domain, subdomain, keywords, difficulty, description）
- **specification.json**: 结构完全匹配（study_id, participants, design, procedure）
- **ground_truth.json**: 结构完全匹配（validation_criteria.required_tests格式，original_results格式）

### 2. Config类兼容

- 继承`BaseStudyConfig`
- 实现`create_trials()`和`aggregate_results()`
- 注册到`StudyConfigRegistry`
- 可通过`get_study_config()`工厂函数访问

### 3. 文件结构兼容

- `data/studies/study_XXX/`目录结构
- `metadata.json`, `specification.json`, `ground_truth.json`位置
- `materials/`目录（如果需要）

## 实现优先级

1. **Phase 1**: Filter + Review格式（最简实现）
2. **Phase 2**: Extractor + Review格式
3. **Phase 3**: JSON生成器（确保兼容性）
4. **Phase 4**: Config生成器（统一接口）

## 技术选择

- **LLM**: 复用`GeminiClient`（已有）
- **PDF处理**: 复用`DocumentLoader`（已有）
- **Review解析**: 简单markdown解析（不需要复杂库）
- **代码生成**: 字符串模板（Jinja2或f-string）

## CLI接口（极简）

```bash
# 运行阶段1（从当前目录的PDF文件）
python generation_pipeline/run.py --stage 1

# 运行阶段2（基于阶段1的结果）
python generation_pipeline/run.py --stage 2

# Refine阶段1
python generation_pipeline/run.py --stage 1 --refine

# Refine阶段2
python generation_pipeline/run.py --stage 2 --refine
```

**设计**:
- `--stage 1|2`: 指定运行阶段（必需）
- `--refine`: 可选，表示refine当前阶段
- PDF文件：从当前目录自动检测，或通过配置文件指定
- Paper ID：从输出文件名或配置文件自动推断

## 验证兼容性

生成后需要验证：
1. JSON文件可以通过`Study.load()`加载
2. Config类可以通过`get_study_config()`创建
3. 可以通过`run_full_benchmark.py`运行
4. 可以通过`Scorer`评分

