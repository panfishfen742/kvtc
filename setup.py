from setuptools import find_packages, setup


setup(
    name="kvtc",
    version="0.1.0",
    description="First open-source implementation of KV Cache Transform Coding (KVTC).",
    packages=find_packages(),
    py_modules=[],
    install_requires=["torch", "numpy", "transformers"],
    extras_require={"dev": ["pytest"]},
    python_requires=">=3.10",
)
