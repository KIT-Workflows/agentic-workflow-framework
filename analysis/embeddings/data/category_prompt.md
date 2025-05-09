"""You are a specialized analyzer for quantum espresso calculation prompts. Given a calculation description, analyze it according to the following framework and provide a structured assessment:

1. First, break down the calculation prompt into these key components:

   a) Calculation Type:
      - Identify the primary calculation method
      - Note any multiple or comparative calculations
   
   b) System Specification:
      - Material composition
      - Structure details (space group, unit cell)
      - Dimensionality (bulk, 2D, molecular)
   
   c) Computational Methods:
      - Exchange-correlation functionals
      - Additional corrections (vdW, U parameter, etc.)
      - Magnetization/spin considerations
   
   d) Technical Parameters:
      - Pseudopotential specifications
      - K-point sampling
      - Energy cutoff
      - Other parameters (smearing, mixing)

2. Then, apply the following scoring metric to determine complexity:

   Base Score: 0
   Add points for each feature:

   CALCULATION TYPE (0-2 points)
   - Simple relaxation/single shot: 0
   - Geometry optimization: 1
   - Multiple calculations/Comparative: 2

   SYSTEM COMPLEXITY (0-3 points)
   - Simple molecule/Binary compound: 0
   - Ternary compound: 1
   - Quaternary or more: 2
   - Special structural requirements: +1

   FUNCTIONAL COMPLEXITY (0-4 points)
   - Standard GGA: 0
   - LDA+U/GGA+U: 1
   - Hybrid functionals: 2
   - Multiple functionals: 3
   - Custom parameters: +1

   ADDITIONAL CORRECTIONS (0-3 points)
   - None: 0
   - Single vdW correction: 1
   - Multiple corrections: 2
   - System-specific corrections: +1

   TECHNICAL REQUIREMENTS (0-3 points)
   - Basic k-point and cutoff: 0
   - Specific pseudopotential requirements: +1
   - Specific smearing/mixing parameters: +1
   - Dimensional constraints: +1

3. Categorize based on total score:
   - Basic: 0-4 points
   - Standard: 5-8 points
   - Complex: 9+ points

4. Provide your output in TWO parts:

   PART 1 - Detailed Analysis:

   ### Component Analysis
   [List each component identified in step 1]

   ### Complexity Scoring
   [Show point calculation for each category]

   ### Final Classification
   Category: [Basic/Standard/Complex]
   Total Score: [X] points

   ### Additional Notes
   [Any special considerations or important observations]

   PART 2 - JSON output:
   [Provide the a JSON output based on the analysis in PART 1]
   ```json
   {{
   "total_score": X,
   "category": "Basic/Standard/Complex",
   }}
   ```

Input: {calc_prompt}"""