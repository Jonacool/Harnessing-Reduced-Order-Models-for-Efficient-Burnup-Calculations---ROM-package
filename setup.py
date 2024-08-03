from setuptools import setup, find_packages

setup(
    name="rom",
    version="0.69",
    packages=find_packages(),
    install_requires=["numpy", "scipy"],
    author="Jonathan Pilgram",
    author_email="jonathanpilgram@gmail.com",
    description="Reduced Order Modelling in a nuclear context",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    # url='https://github.com/yourusername/my_package',  # Replace with your package URL
    # classifiers=[
    #    'Programming Language :: Python :: 3',
    #    'License :: OSI Approved :: MIT License',
    #    'Operating System :: OS Independent',
    # ],
    python_requires=">=3.6",
)
