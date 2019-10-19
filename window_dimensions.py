# try:
#     import win32gui
#
# except ImportError:
#     print('win32gui not installed')
# def callback(hwnd, extra):
#     rect = win32gui.GetWindowRect(hwnd)
#     x = rect[0]
#     y = rect[1]
#     w = rect[2] - x
#     h = rect[3] - y
#     print("Window %s:" % win32gui.GetWindowText(hwnd))
#     print("\tLocation: (%d, %d)" % (x, y))
#     print("\t    Size: (%d, %d)" % (w, h))
#     return [x, y, w, h]
#
# def get_Window_Dimensions():
#     win32gui.EnumWindows(callback, None)
#
# if __name__ == '__main__':
#     get_Window_Dimensions()
import logging
import sys
import time
import pickle as pkl
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

try:
    # def for_Linux():
    #     try:
    #         import wnck
    #     except ImportError:
    #         logging.info("wnck not installed")
    #         wnck = None
    #     if wnck is not None:
    #         screen = wnck.screen_get_default()
    #         screen.force_update()
    #         window = screen.get_active_window()
    #         if window is not None:
    #             pid = window.get_pid()
    #             with open("/proc/{pid}/cmdline".format(pid=pid)) as f:
    #                 active_window_name = f.read()
    #             x, y, w, h = wnck.window.get_client_window_geometry()
    #             print('dimensions: ', x,y,w,h)
    #             return [x, y, w, h]
    #     else:
    #         try:
    #             from gi.repository import Gtk, Wnck
    #             gi = "Installed"
    #         except ImportError:
    #             logging.info("gi.repository not installed")
    #             gi = None
    #         if gi is not None:
    #             Gtk.init([])  # necessary if not using a Gtk.main() loop
    #             screen = Wnck.Screen.get_default()
    #             screen.force_update()  # recommended per Wnck documentation
    #             windows =screen.get_windows()
    #             browser_names = ['Firefox','Mozilla Firefox','Chrome', 'Google Chrome', 'Safari', 'Mss :: Anaconda Cloud - Google Chrome']
    #             for each in windows:
    #                 name = each.get_application().get_name()
    #                 print(name)
    #                 #if name in browser_names:
    #                 if any(substring in name for substring in browser_names):
    #                     browser_window = each
    #                     #browser_window.activate(1)
    #                     # browser_window.set_fullscreen(True)

    #                     x, y, w, h = Wnck.Window.get_geometry(browser_window)
    #                     print(x, y, w, h)	
    #                     print(name)
    #                 else:
    #                     each.minimize()
                
    #             dim = []
    #             with open("win_dim.pkl", "rb") as f:
    #                 dim = pkl.load(f)
    #                 print("DIMENSIONSSSSSS:::::", dim)
            
    #             print("DIMENSIONSSSSSS:::::", dim)
    #             return dim

    def for_Linux():
        import gi

        gi.require_version('Gdk', '3.0')
        gi.require_version('Gtk', '3.0')
        from gi.repository import Gtk, Gdk

        # Replace w with the GtkWindow of your application
        w = Gtk.Window()
        # Get the screen from the GtkWindow
        s = w.get_screen()
        # Using the screen of the Window, the monitor it's on can be identified
        m = s.get_monitor_at_window(s.get_active_window())
        # Then get the geometry of that monitor
        monitor = s.get_monitor_geometry(m)
        # This is an example output
        print("Height: %s, Width: %s" % (monitor.height, monitor.width))

        return [0,0,monitor.height, monitor.width]
except Exception as e:
    print(e)

def callback(hwnd, extra):
    global dimensions
    try:
        import win32gui
        import win32con 
    except ImportError:
        print('win32gui not installed')
    title = win32gui.GetWindowText(hwnd)
    browser_names = ['Firefox','Mozilla Firefox','Chrome', 'Google Chrome', 'Safari']
    for each in browser_names:
        if each in title:
            hwnd = win32gui.FindWindow(None, title)
            win32gui.SetForegroundWindow(hwnd)
            dimensions = win32gui.GetWindowRect(hwnd)
            print(win32gui.GetForegroundWindow())
            # win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
            rect = win32gui.GetWindowRect(hwnd)
            x = rect[0]
            y = rect[1]
            w = rect[2] - x
            h = rect[3] - y
            print("Window %s:" % title)
            print("\tLocation: (%d, %d)" % (x, y))
            print("\t    Size: (%d, %d)" % (w, h))
            # return [x, y, w, h]
            dimensions = [x, y, w, h]
            break         
def for_Windows():
    global dimensions
    try:
        import win32gui
    except ImportError:
        print('win32gui not installed')
    win32gui.EnumWindows(callback, None)
    return dimensions


def for_Mac():
    from subprocess import Popen, PIPE

    # http://stackoverflow.com/a/373310/562769
    # from AppKit import NSWorkspace, NSScreen, NSWindow
    # active_window_name = (NSWorkspace.sharedWorkspace()
    #     .activeApplication())
    # print('ACTIVE', active_window_name)
    # windows = (NSWorkspace.sharedWorkspace().runningApplications())
    # print(windows)
    # from AppKit import NSScreen
    #
    # frame = NSScreen.frame
    # convertRectToBacking(frame)
    #
    #

    windowSize = '''
                    tell application "Chrome"
                    get the bounds of the window 1
                    end tell
                    '''
    proc = Popen(['osascript', '-'], stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    wSize, error = proc.communicate(windowSize)
    wSize = wSize.replace(', ', 'x')
    print('got dimensions yo')
    dimensions = list(map(int, wSize.split('x')))
    return dimensions

def get_active_window_dimensions():
    """
    Get the currently active window.

    Returns
    -------
    string :
        Name of the currently active window.
    """
    import sys
    active_window_name = None
    if sys.platform in ['linux', 'linux2']:
        # Alternatives: http://unix.stackexchange.com/q/38867/4784
        dimensions = for_Linux()
        return dimensions
    elif sys.platform in ['Windows', 'win32', 'cygwin']:
        # http://stackoverflow.com/a/608814/562769
        try:
            import win32gui
        except ImportError:
          print('yolo') 
          return 
        window = win32gui.GetForegroundWindow()
        active_window_name = win32gui.GetWindowText(window)
        dimensions = for_Windows()
        return dimensions
    elif sys.platform in ['Mac', 'darwin', 'os2', 'os2emx']:
        dimensions = for_Mac()
        return dimensions

    else:
        print("sys.platform={platform} is unknown. Please report."
              .format(platform=sys.platform))
        print(sys.version)
    return active_window_name

# print("Active window: %s" % str(get_active_window_dimensions()))
