import os
from glob import glob
from setuptools import setup

package_name = 'freebrain_description'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*')),
        (os.path.join('share', package_name, 'mjcf'), glob('mjcf/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='joo',
    maintainer_email='joo@todo.todo',
    description='URDF and mesh descriptions for FreeBrain (OpenManipulator-X)',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [],
    },
)
