# CADPrompt slice

100 of 200 items, seed 20260915, dimensioned prompts, DeepCAD units (not mm: card scores are unit-normalised, and every acceptance entry carries `normalized: true`). Source https://github.com/Kamel773/CAD_Code_Generation (no licence file; evaluation only, not redistributed).

Prompts are the upstream `..._with_specific_measurements.txt` text with the leading code-generation instruction ("Write Python code using CADQuery to ") stripped and the next letter capitalised. The rest of the sentence is verbatim: the suite asks for a part, not for a CadQuery program.
