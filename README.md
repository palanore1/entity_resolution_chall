# Entity Resolution Optimization Journey

This document outlines the performance optimization journey for the entity resolution script, explaining the challenges faced and the solutions implemented to achieve reasonable processing times.

## Initial Challenge

The original implementation faced significant performance issues:
- Processing time exceeded 5 minutes
- Exhaustive pairwise comparisons within blocks
- Inefficient blocking strategy creating too many large blocks
- No early stopping mechanisms

## Optimization Journey

### Phase 1: Initial Optimizations

#### Blocking Strategy Improvements
- Removed industry/sector blocking which created too many large blocks
- Increased minimum word length for first word blocking from 2 to 3 characters
- Increased phone number prefix length from 3 to 4 digits
- Added maximum block size limit of 100 companies
- Filtered out blocks with only one company

#### Comparison Strategy Improvements
- Reduced max_comparisons from 100 to 10 per company
- Added early stopping when finding a good match (similarity >= threshold)
- Added maximum match limit of 1000 matches
- Implemented quick check for exact matches before computing similarity
- Added processed_pairs set to avoid duplicate comparisons

#### Similarity Threshold Adjustments
- Increased similarity threshold to 0.90
- Added early stopping when similarity threshold is reached
- Implemented weighted combination of similarity metrics based on name length

### Phase 2: Fine-tuning for Balance

After the initial optimizations, the script ran almost instantly, which raised concerns about potential under-matching. We made the following adjustments to find a better balance between speed and accuracy:

#### Adjusted Parameters
1. **Similarity Threshold**
   - Reduced from 0.90 to 0.88
   - Reasoning: The higher threshold was too strict, potentially missing valid matches
   - Impact: Slightly increased false positives but catches more valid matches

2. **Comparison Limits**
   - Increased max_comparisons from 10 to 25 per company
   - Increased max_matches from 1000 to 2000
   - Reasoning: Previous limits were too restrictive, potentially missing good matches
   - Impact: More thorough comparison while maintaining reasonable performance

3. **Blocking Strategy Refinements**
   - Reduced minimum word length from 3 to 2 characters
   - Reduced phone prefix length from 4 to 3 digits
   - Increased maximum block size from 100 to 200 companies
   - Reasoning: Previous blocking was too aggressive, creating too few blocks
   - Impact: More focused blocks while still maintaining manageable sizes

## Performance Results

### Before Any Optimizations
- Processing time: > 5 minutes
- No early stopping
- Exhaustive comparisons
- Large blocks causing quadratic growth in comparisons

### After Phase 1 Optimizations
- Processing time: Almost instant
- Early stopping mechanisms in place
- Limited comparisons per company
- Controlled block sizes
- Efficient blocking strategy

### After Phase 2 Fine-tuning
- Processing time: ~2-3 minutes
- Better balance between speed and accuracy
- More thorough matching while maintaining reasonable performance
- Improved handling of edge cases

## Key Learnings

1. **Blocking Strategy**
   - Quality of blocks is more important than quantity
   - Large blocks can significantly impact performance
   - Need to balance between block size and number of blocks
   - Too aggressive blocking can lead to under-matching

2. **Comparison Strategy**
   - Not all comparisons are necessary
   - Early stopping can significantly reduce processing time
   - Duplicate comparisons should be avoided
   - Too few comparisons can lead to missing valid matches

3. **Threshold Selection**
   - Higher thresholds can reduce false positives but might miss valid matches
   - Early stopping based on thresholds can improve performance
   - Different metrics might be needed for different name lengths
   - Need to balance between precision and recall

## Future Improvements

Potential areas for further optimization:
1. Parallel processing for independent blocks
2. More sophisticated blocking strategies
3. Caching of similarity computations
4. Dynamic adjustment of comparison limits based on block size
5. Implementation of approximate nearest neighbors algorithms

## Usage

To run the optimized script:
```bash
python entity_resolution.py
```

The script will:
1. Load and preprocess the dataset
2. Create optimized blocks
3. Find matches using the improved comparison strategy
4. Output results sorted by similarity score

## Dependencies

Required Python packages:
- pandas
- numpy
- scikit-learn
- scipy
- jellyfish
- unidecode 