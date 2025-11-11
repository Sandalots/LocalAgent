# 🔧 Bug Fixes Applied

## Issues Found & Fixed:

### 1. ✅ **Dependencies Parsing Bug**
**Problem**: Comments in requirements.txt breaking pip install
```
faiss-cpu>=1.7.4  # or faiss-gpu for GPU support
```

**Fix**: Updated `_extract_dependencies()` to properly strip inline comments

### 2. ✅ **Entry Points Detection Bug**  
**Problem**: Found 113 entry points (including venv/site-packages files)

**Fix**: 
- Only search root directory (not recursive)
- Filter out venv, site-packages, hidden directories
- Use glob patterns: `main*.py`, `train*.py`, `run*.py`

### 3. ✅ **README-Based Execution**
**Added**: Parse README for python commands and prioritize them
- Looks for: `python3 main_local_all_new.py`
- Extracts arguments (like `--oracle`)
- Runs README scripts before generic entry points

### 4. ✅ **LLM Extraction Improvements**
**Problem**: Methodology and experiments returned empty (0 chars)

**Fix**:
- Better prompting for JSON extraction
- Added fallback regex-based extraction
- More explicit instructions to LLM

## 🧪 Test Again

```bash
source .venv/bin/activate
python run.py
```

### Expected Behavior:
1. ✅ Find `main_local_all_new.py` as entry point
2. ✅ Parse dependencies correctly (no comments)
3. ✅ Extract paper sections with LLM
4. ✅ Run the correct experiment script from README

## 📊 What Should Happen Now:

```
✓ Found paper: 300_Decontextualization_Everyw.pdf
✓ Found code directory: ...
✓ Ollama is running
✓ Extracted abstract (XXX chars)
✓ Extracted methodology (XXX chars)  ← Should have content now
✓ Extracted experiments (XXX chars)  ← Should have content now
✓ Analyzed codebase (language: python)
✓ Found 1 potential entry points        ← Should be 1, not 113!
✓ Found 29 dependencies
Setting up experiment environment...
✓ Dependencies installed successfully   ← Should work now!
Found priority script from README: main_local_all_new.py
Running priority script: main_local_all_new.py
```

Try it now! 🚀
