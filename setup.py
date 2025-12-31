from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'mppi_torch'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'weights'), glob('weights/*')),
    ],
    install_requires=['setuptools', 'torch', 'numpy'],
    zip_safe=True,
    maintainer='kyo',
    maintainer_email='s21c1135sc@s.chibakoudai.jp',
    description='MPPI-based path tracking controller using PyTorch',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mppi_node = mppi_torch.mppi_node:main',
        ],
    },
)
