from setuptools import setup, find_packages

setup(

    name="inference-kernels",

    version="0.1.0",

    description="Fused Triton kernels for LLM inference (RMSNorm, SwiGLU, RoPE, Softmax)",

    author="RAGul  .",

    package_dir={"": "python"},

    packages=find_packages(where="python"),

    install_requires=[

        "torch>=2.0.0",

        "triton>=2.0.0",

    ],

    python_requires=">=3.9",

)
