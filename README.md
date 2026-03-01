# 2026 Hackathon

This is the repository for the 2026 Hackathon project.

Semantic map workflow (v1):
- Open `public/map-editor.html`
- Paint layers: `walkable`, `exits`, `doors`, `stairs`
- Export JSON (`pyroplan-map.json`)
- Run simulation with that JSON as the map file
  - CLI example: `python main.py --map pyroplan-map.json --out evac.mp4 --fire random`

Refrences:

- Pleaae refer to license.txt for licensure information
- Please refer to legal_disclaimer.txt for legal information
