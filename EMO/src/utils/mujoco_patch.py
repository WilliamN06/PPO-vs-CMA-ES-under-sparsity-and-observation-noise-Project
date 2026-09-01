import os
import sys


def apply_mujoco_patch():
    """Apply MuJoCo patch for headless rendering."""
    # Headless rendering, no display on compute nodes
    os.environ['MUJOCO_GL'] = 'osmesa'
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    os.environ['EGL_PLATFORM'] = 'surfaceless'

    # Path is read from env so isca and runpod can point at different installs
    mujoco_path = os.environ.get('MUJOCO_PATH', '/lustre/projects/Research_Project-T132310/mujoco/mujoco-3.2.5')
    os.environ['MUJOCO_PATH'] = mujoco_path
    
    # Add MuJoCo lib to LD_LIBRARY_PATH
    lib_path = mujoco_path + '/lib'
    if lib_path not in os.environ.get('LD_LIBRARY_PATH', ''):
        os.environ['LD_LIBRARY_PATH'] = lib_path + ':' + os.environ.get('LD_LIBRARY_PATH', '')

    # Also set for mujoco-py compatibility
    os.environ['MUJOCO_PY_MUJOCO_PATH'] = mujoco_path

    # Mock OpenGL so import does not blow up without a display
    class MockGL:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    # Mock OpenGL modules
    for mod in ['OpenGL', 'OpenGL.GL', 'OpenGL.GL.VERSION', 'OpenGL.raw',
                'OpenGL.raw.GL', 'OpenGL.raw.GL.VERSION', 'OpenGL.GL.VERSION.GL_1_1',
                'OpenGL.GL.VERSION.GL_1_2', 'OpenGL.GL.VERSION.GL_1_3']:
        sys.modules[mod] = MockGL()

    # Also mock GLUT if present
    if 'OpenGL.GLUT' not in sys.modules:
        sys.modules['OpenGL.GLUT'] = MockGL()

    try:
        import mujoco
        print(f"mujoco loaded ok, path={mujoco_path}")
        return True
    except ImportError as e:
        print(f"mujoco import failed: {e}")
        return False
