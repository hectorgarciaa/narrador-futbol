def normalize_detection_class_name(class_name):
    token = str(class_name or "").strip().lower()
    aliases = {
        "person": "player",
        "persons": "player",
        "people": "player",
        "human": "player",
        "player": "player",
        "players": "player",
        "goalkeeper": "goalkeeper",
        "goalkeepers": "goalkeeper",
        "gk": "goalkeeper",
        "keeper": "goalkeeper",
        "goalie": "goalkeeper",
        "referee": "referee",
        "referees": "referee",
        "ref": "referee",
        "refs": "referee",
        "arbitro": "referee",
        "arbitros": "referee",
        "árbitro": "referee",
        "árbitros": "referee",
        "ball": "ball",
        "balls": "ball",
        "sports ball": "ball",
        "sports balls": "ball",
    }
    return aliases.get(token, "")