"""A Role Playing Game in the Cyberpunk RED Universe"""
import cmd
import os
import sys
import shelve
import time
from NeonCore.game_maps import Map

class ActionManager(cmd.Cmd):
    """cli, displays character stats/skills, quits the game"""
    intro = r"""     ᐸ ソ ╱> /Ξ /≥ /> // /𐑘/ /ᐸ
                      ‾
   …   ˛⁄⁔      ˛⁔     ⁔   _  ¸¸
  (˙}  \(∞l   ,, {˚)/ ¸{=}˛ |\\(˚}
 /(\)╲  `••\˛_ \/(⎔◊𐑘 (\+/)  \∏(p)]
 \ᢘ╦╤═÷- Y¸∆     ¸U˛   \Ξ˛\   ´¸v˛|
  7˘ 𐑘 ¸⁄∫𐑘      [][]   7 𐑘 `   [ ]´
  ]  ]  / |      [ [   ]  ]    { }
  l  L ∫  l      ɺ[ɺ]  l  L    ɺ L 
    ⌁help⌁   give me the 411
"""
    prompt = 'ᐸ/> '
    ruler = '⌁'
    doc_header = "Recorded jive (type help <jargon>):"

    def __init__(self, char_mngr, cmd_mngr):
        super().__init__()
        self.char_mngr = char_mngr
        self.cmd_mngr = cmd_mngr
        self.game_map = None 
        self.game_state = 'choose_character'

    def start_game(self):
        os.system('clear')
        self.prompt = '(choose_character) '
        #print(self.cmd_mngr.commands)
        print(self.cmd_mngr.do_choose_character)
        self.cmdloop()

    def completenames(self, text, *ignored):
        cmds = super().completenames(text, *ignored)
        check_cmd = self.cmd_mngr.get_check_command(self)
        if check_cmd:
            cmds += [c for c in check_cmd if c.startswith(text)]
            #print(cmds)
            #[c for c in check_cmd.get_available_commands() if 
            #         c.startswith(text)]
        return cmds

    def get_available_commands(self):
        return [name[3:] for name in dir(self) if name.startswith("do_")]

    def do_shell(self, arg):
        """ Shell commands can be added here prefixed with !"""
        os.system('clear')

   # def postcmd(self, stop, line):
   #     # Get the number of rows in the output
   #     rows, _ = os.popen('stty size', 'r').read().split()
   #     # Move the cursor to the correct position
   #     print(f'\033[{int(rows)}H')
   #     return stop
   #     # Get the size of the terminal window
   #     rows, cols = self.stdscr.getmaxyx()
   #     # Move the cursor to the 24th row (0-indexed)
   #     self.stdscr.move(24, 0)
   #     # Refresh the terminal window
   #     self.stdscr.refresh()
   #     return cmd.Cmd.postcmd(self, stop, line)

    def default(self, line):
        print("WTF dat mean, ain't no command like dat")

    def do_quit(self, arg):
        """Exits Cyberpunk RED"""
        wprint("Catch you on the flip side, choombatta. Keep your chrome "
               "polished and your guns loaded, "
               "the neon jungle ain't no walk in the park.")
        # Open database, create if it doesn't already exist
        with shelve.open('timestamp') as dbase:
            # Save data to the database>
            dbase['timestamp'] = time.time()
        sys.exit()

    def do_move(self, args):
        """Move player in the specified direction"""
        my_map = Map(self.char_mngr.player, 
                     self.char_mngr.npcs)
        my_map.move()

## Open a shelve in read mode
#with shelve.open('timestamp', 'r') as db:
#    # Load the timestamp
#    timestamp = dbase['timestamp']
#print(timestamp)

