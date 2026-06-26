"""
Shared bot/moderator filtering used by every pipeline step, so the exclusion is
identical everywhere.
"""

KNOWN_BOTS = {
    "AutoModerator", "sneakpeek_bot", "reddit_bot", "BotDefense",
    "[deleted]", "[removed]",
}


def is_bot(author: str) -> bool:
    """
    True for accounts that are not regular human participants: the curated list
    above, subreddit moderator-team accounts (handles ending in 'ModTeam'), and
    accounts whose handle ends in 'bot' -- but not 'robot', which is common in
    human handles.
    """
    if not author or author in KNOWN_BOTS:
        return True
    if author.endswith("ModTeam"):
        return True
    low = author.lower()
    if low.endswith("bot") and not low.endswith("robot"):
        return True
    return False
