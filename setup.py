import setuptools

with open('README.md', encoding='utf-8') as fid:
    long_description = fid.read()

setuptools.setup(
    name='pyqet', #this is NOT python-module name, one package could include mutiple modules
    version='0.0.0',
    package_dir={'':'python'},
    packages=setuptools.find_packages('python'),
    description='a python package for quantum entanglement detection',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'sympy',
        'scipy',
        'torch',
        'cvxpy', # macos user may need to install it by conda or other package manager instead of pip
        'tqdm',
        'cvxpylayers',
        'matplotlib',
        'pytest',
        'torch-wrapper @ git+https://github.com/husisy/torch-wrapper.git',
    ],
)
