### PyTorch Tips
1. When working with tensors, always use PyTorch’s operators (such as `torch.exp`, `torch.cos`, `torch.sqrt`, ...) to ensure compatibility and optimal performance.
2. In PyTorch, operator input arguments must be tensors, not floats. To handle float numbers, convert float numbers to scalar tensor. For examples, torch.sqrt(torch.tensor(2 * torch.pi)) instead of torch.sqrt(2 * torch.pi).

### Code Requirements

1. The programming language is always python.
2. Annotate the size of the tensor as comment after each tensor operation. For example, `# (B, 3, 3)`.
3. The only library allowed is PyTorch. Follow the examples provided by the user and check the PyTorch documentation to learn how to use PyTorch.
4. Separate the code into parameters that can be tuned with differentiable optimization and the symbolic expression represented by PyTorch code. Define them respectively in the `__init__` function and the `forward` function.
5. The proposed code must strictly follow the structure and function signatures below:

```python
{code}
```

### Solution Requirements

1. Analyze step-by-step what the potential problem is in the previous iterations based on the feedback. Think about why the results from previous iterations mismatched with the ground truth. Do not give advice about how to optimize. Focus on the formulation of the scientific equation. Start this section with "### Analysis". Analyze all iterations individually, and start the subsection for each iteration with "#### Iteration N", where N stands for the index. Remember to analyze every iteration in the history.
2. Think step-by-step what you need to do in this iteration. Think about what is needed to improve performance. If the analysis suggests specific functional forms or constraints, think about how these will be incorporated into the symbolic equation. Think about how to separate your algorithm into a continuous parameter part and a symbolic expression model part. Describe your plan in pseudo-code, written out in great detail. Remember to update the default values of the trainable parameters based on previous optimizations. Start this section with "### Step-by-Step Plan".
3. Output the code in a single code block "```python ... ```" with detailed comments in the code block. Do not add any trailing comments before or after the code block. Start this section with "### Code".