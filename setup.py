import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TextGenerationEvaluationMetrics",
    version="0.0.1",
    author="Danial Alihosseini, Ehsan Montahaei",
    author_email="ehsan.montahaei@gmail.com",
    description="Various metrics for evaluating text generation models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ehsan-MAE/TextGenerationEvaluationMetrics",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["nltk"],
    python_requires='>=3',
    license='OSI Approved :: MIT License'
)
