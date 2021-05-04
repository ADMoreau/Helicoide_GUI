import gi
gi.require_version('Gtk', '3.0')                                                                                                          
from gi.repository import Gtk                                                                                                             
gi.require_version('GdkX11', '3.0')                                                                                                       
from gi.repository import GdkX11    
import vlc


class ApplicationWindow(Gtk.Window):
    """
    class that opens a gtk window to play video inside of vlc window
    """
    
    def __init__(self, MRL):
        Gtk.Window.__init__(self, title="")
        self.player_paused=False
        self.is_player_active = False
        self.connect("destroy",Gtk.main_quit)
        self.set_decorated(False)
        self.MRL = MRL
           
    def show(self):
        self.show_all()
        
    def setup_objects_and_events(self):
        """
        play, stop, pause buttons
        """
        self.draw_area = Gtk.DrawingArea()
        self.draw_area.set_size_request(800,460)
        
        self.draw_area.connect("realize",self._realized)
        
        self.hbox = Gtk.Box(spacing=6)
        
        self.vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.add(self.vbox)
        self.vbox.pack_start(self.draw_area, True, True, 0)
        self.vbox.pack_start(self.hbox, False, False, 0)
        
    def stop_player(self, widget, data=None):
        self.player.stop()
        self.is_player_active = False
        self.destroy()
        
    def _realized(self, widget, data=None):
        self.vlcInstance = vlc.Instance("--no-xlib")
        self.player = self.vlcInstance.media_player_new()
        win_id = widget.get_window().get_xid()
        self.player.set_xwindow(win_id)
        self.player.set_mrl(self.MRL)
        self.player.play()
        self.is_player_active = True
