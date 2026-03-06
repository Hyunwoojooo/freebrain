from setuptools import setup

package_name = 'freebrain_safety'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/safety_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='joo',
    maintainer_email='joo@todo.todo',
    description='FreeBrain safety package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'safety_node = freebrain_safety.ros_node:main',
        ],
    },
)
