import os
import re

def generate_top_half():
    input_filename = r"C:\Users\willi\Downloads\Cobra_PLA_14h55m.gcode"
    output_filename = "Cobra_Top_Half.gcode"
    
    target_z = 69.600
    target_line_num = 3028377  
    
    if not os.path.exists(input_filename):
        print(f"Error: '{input_filename}' not found.")
        return

    print(f"Reading '{input_filename}'...")
    with open(input_filename, 'r') as f:
        lines = f.readlines()

    cut_index = target_line_num - 1

    # Look ahead to find the starting X/Y coordinates
    target_x, target_y = "100.00", "100.00"
    for line in lines[cut_index:]:
        if "G1" in line or "G0" in line:
            match_x = re.search(r"X([\d\.]+)", line)
            match_y = re.search(r"Y([\d\.]+)", line)
            if match_x and match_y:
                target_x = match_x.group(1)
                target_y = match_y.group(1)
                break

    top_half_header = f"""\
; === TOP HALF STARTUP SEQUENCE ===
M201 X2500 Y2500 Z500 E5000   
M203 X250 Y250 Z5 E40         
M204 P2500 R500 T2500 
M205 X10.00 Y10.00 Z0.40 E5.00 

M220 S100 ; Reset Feedrate 
M221 S100 ; Reset Flowrate 

; Standard Heating
M140 S65  
M104 S220 
M190 S65  
M109 S220 

G90 ; Absolute positioning
M83 ; Relative extrusion

; Standard Auto-Home on the bare bed
G28 X Y Z

; Lift nozzle to a physical height of 5.0mm
G1 Z5.0 F3000 

; --- THE Z-OFFSET TRICK ---
; Tell the printer that this physical 5.0mm is actually {target_z + 5.0}!
G92 Z{target_z + 5.0:.3f}

; Move to the starting X/Y coordinates while safely hovering 5mm above the bed
G1 X{target_x} Y{target_y} F6000

; Drop down to the virtual layer height of {target_z}. 
; (Because of the trick above, this physically drops the nozzle to exactly 0.0mm on the glass!)
G1 Z{target_z:.3f} F1200
; === END STARTUP SEQUENCE ===

; === RESUMING TOOLPATH ON THE BED ===
"""

    print(f"Slicing off everything before line {target_line_num}...")
    
    with open(output_filename, 'w') as out_f:
        out_f.write(top_half_header)
        for line in lines[cut_index:]:
            out_f.write(line)

    print(f"\nSuccess! File saved as '{output_filename}'")
    print("Clear your bed, load this file normally, and hit Print.")

if __name__ == "__main__":
    generate_top_half()