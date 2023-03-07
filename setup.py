from setuptools import setup

setup(
    name='model_based_rl',
    version='0.1',
    url='https://git.ias.informatik.tu-darmstadt.de/palenicek/value_expansion',

    author='Daniel Palenicek',
    author_email='daniel.palenicek@tu-darmstadt.de',

    packages=['model_based_rl',],

    license='MIT',

    install_requires=[
        'jax==0.3.5',
        'jaxlib@https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.5+cuda11.cudnn805-cp37-none-manylinux2010_x86_64.whl',
        'wandb',
        'ml_collections==0.1.0',
        'dm-haiku==0.0.5',
        'flax==0.4.1'
    ],
)
