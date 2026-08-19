"""
Neuroplasticity Explorer — region catalog + hybrid answer engine.

Hybrid design:
  1. Curated knowledge base answers popular topics instantly, at zero cost,
     with hand-checked confidence tags.
  2. Claude (structured JSON output) handles anything else, constrained to the
     same region vocabulary so the 3D scene can always render the answer.

The region list here is the single source of truth — the frontend fetches it
from /api/brain/regions so the meshes and the answers can never drift apart.
"""
import os
import re
import json

# ── Region catalog ───────────────────────────────────────────────────────
# center/radii are in a stylized brain space (mm-ish), x = left(-)/right(+),
# y = inferior(-)/superior(+), z = posterior(-)/anterior(+).
# `bilateral` regions are mirrored across the midline by the renderer.
REGIONS = [
    {"id": "prefrontal_cortex", "label": "Prefrontal cortex", "bilateral": True,
     "center": [28, 22, 58], "radii": [26, 26, 28], "hue": 265,
     "blurb": "Planning, sustained attention, impulse control."},
    {"id": "orbitofrontal_cortex", "label": "Orbitofrontal cortex", "bilateral": True,
     "center": [22, -14, 55], "radii": [20, 14, 22], "hue": 285,
     "blurb": "Value, reward appraisal, regret."},
    {"id": "anterior_cingulate", "label": "Anterior cingulate", "bilateral": False,
     "center": [0, 18, 28], "radii": [10, 22, 30], "hue": 200,
     "blurb": "Conflict monitoring, effort, error detection."},
    {"id": "motor_cortex", "label": "Motor cortex", "bilateral": True,
     "center": [30, 44, 8], "radii": [22, 16, 14], "hue": 175,
     "blurb": "Voluntary movement, skill encoding."},
    {"id": "somatosensory_cortex", "label": "Somatosensory cortex", "bilateral": True,
     "center": [32, 42, -10], "radii": [22, 16, 14], "hue": 160,
     "blurb": "Touch, body mapping."},
    {"id": "parietal_cortex", "label": "Parietal cortex", "bilateral": True,
     "center": [34, 34, -38], "radii": [24, 22, 24], "hue": 190,
     "blurb": "Spatial attention, integration across senses."},
    {"id": "visual_cortex", "label": "Visual cortex", "bilateral": True,
     "center": [22, 6, -72], "radii": [24, 24, 20], "hue": 220,
     "blurb": "Visual processing, pattern and motion."},
    {"id": "temporal_auditory", "label": "Temporal / auditory cortex", "bilateral": True,
     "center": [52, -8, 2], "radii": [16, 20, 34], "hue": 310,
     "blurb": "Sound, language, semantic memory."},
    {"id": "hippocampus", "label": "Hippocampus", "bilateral": True,
     "center": [26, -14, -8], "radii": [8, 8, 24], "hue": 95,
     "blurb": "Encoding new memories, consolidation, neurogenesis."},
    {"id": "amygdala", "label": "Amygdala", "bilateral": True,
     "center": [24, -12, 14], "radii": [9, 8, 9], "hue": 5,
     "blurb": "Threat salience, emotional tagging."},
    {"id": "striatum", "label": "Striatum / nucleus accumbens", "bilateral": True,
     "center": [16, 2, 16], "radii": [11, 14, 20], "hue": 35,
     "blurb": "Dopamine reward prediction, habit formation."},
    {"id": "thalamus", "label": "Thalamus", "bilateral": True,
     "center": [9, 4, -6], "radii": [9, 11, 14], "hue": 50,
     "blurb": "Relay hub, arousal and sensory gating."},
    {"id": "insula", "label": "Insula", "bilateral": True,
     "center": [38, 2, 12], "radii": [7, 18, 20], "hue": 330,
     "blurb": "Interoception, craving, bodily awareness."},
    {"id": "cerebellum", "label": "Cerebellum", "bilateral": True,
     "center": [26, -34, -58], "radii": [26, 20, 24], "hue": 140,
     "blurb": "Timing, coordination, procedural learning."},
    {"id": "brainstem", "label": "Brainstem", "bilateral": False,
     "center": [0, -30, -16], "radii": [11, 30, 13], "hue": 20,
     "blurb": "Arousal, sleep-wake, autonomic control."},
]

REGION_IDS = [r["id"] for r in REGIONS]
CONFIDENCE_LEVELS = ["established", "emerging", "hypothesized"]


# ── Curated knowledge base ───────────────────────────────────────────────
# Each finding: region, effect, mechanism, severity 1-5, confidence, direction.
def _f(region, effect, mechanism, severity, confidence, direction="impair"):
    return {"region": region, "effect": effect, "mechanism": mechanism,
            "severity": severity, "confidence": confidence, "direction": direction}


CURATED = [
    {
        "keywords": ["short form", "short-form", "shorts", "reels", "tiktok", "doomscroll",
                     "doom scroll", "scrolling", "infinite scroll", "swipe", "feed"],
        "title": "Excessive short-form video",
        "summary": "Rapid novelty-reward cycles train the brain for fast switching and "
                   "shallow encoding. The research is real but young — most findings are "
                   "correlational, and 'dopamine damage' framing is not supported.",
        "findings": [
            _f("striatum",
               "Reward circuitry adapts to very short novelty cycles",
               "Each swipe is a variable-ratio reward. Dopamine signals prediction error, "
               "so unpredictable payoff on a short loop is the strongest reinforcer known — "
               "the same schedule slot machines use.", 5, "established"),
            _f("prefrontal_cortex",
               "Sustained-attention circuits get less exercise",
               "Attention is use-dependent. Content that never requires holding a thread for "
               "more than a minute under-trains the top-down control that long-form focus needs; "
               "task-switch costs rise.", 4, "emerging"),
            _f("anterior_cingulate",
               "Lower tolerance for effortful, low-stimulation tasks",
               "Effort valuation shifts when a zero-effort reward is always one swipe away, "
               "making ordinary cognitive work feel disproportionately costly.", 3, "emerging"),
            _f("hippocampus",
               "Shallow encoding — little consolidates into memory",
               "Consolidation needs elaborative processing and gaps between inputs. Back-to-back "
               "clips leave no room for either, so most of it is never encoded.", 3, "emerging"),
            _f("visual_cortex",
               "Habituation to high-intensity editing pace",
               "Fast cuts and dense stimulation raise the baseline of what registers as "
               "engaging, so slower media feels flat by comparison.", 2, "hypothesized"),
        ],
        "note": "Evidence is mostly cross-sectional: heavy use correlates with worse "
                "sustained-attention scores, but causal direction is not settled — people "
                "with lower baseline attention may also scroll more. The good news is the "
                "same plasticity runs in reverse; the effects are not fixed.",
    },
    {
        "keywords": ["chronic stress", "stressed", "stress", "cortisol", "burnout", "overwhelm"],
        "title": "Chronic stress",
        "summary": "Sustained cortisol reshapes the balance between threat detection and "
                   "top-down control — one of the best-replicated findings in the field.",
        "findings": [
            _f("amygdala", "Threat detection becomes hyper-responsive",
               "Chronic glucocorticoid exposure drives dendritic growth in the amygdala, "
               "lowering the threshold at which things register as threatening.", 5, "established"),
            _f("hippocampus", "Impaired memory encoding; suppressed neurogenesis",
               "The hippocampus is dense with cortisol receptors. Prolonged exposure suppresses "
               "new neuron growth and can shrink dendritic branching.", 4, "established"),
            _f("prefrontal_cortex", "Weakened top-down regulation",
               "Prefrontal control over the amygdala degrades under sustained stress, so "
               "emotional responses are harder to modulate.", 4, "established"),
            _f("insula", "Heightened interoceptive alarm signals",
               "Bodily-state monitoring amplifies, feeding the sense of unease.", 2, "emerging"),
        ],
        "note": "Much of this is reversible — hippocampal volume recovers with stress "
                "reduction, sleep, and exercise.",
    },
    {
        "keywords": ["sleep", "sleep deprivation", "insomnia", "no sleep", "all nighter",
                     "not sleeping", "tired"],
        "title": "Sleep deprivation",
        "summary": "Sleep is when consolidation and metabolic clearance happen. Losing it "
                   "degrades nearly every system, and the effects compound.",
        "findings": [
            _f("hippocampus", "Memory consolidation fails",
               "Sharp-wave ripples during deep sleep replay the day's experience into cortex. "
               "No deep sleep, no transfer.", 5, "established"),
            _f("prefrontal_cortex", "Attention, judgment, and impulse control degrade",
               "Prefrontal cortex is the most sleep-sensitive region — it shows measurable "
               "deficits after a single night.", 5, "established"),
            _f("amygdala", "Emotional reactivity amplifies sharply",
               "Prefrontal-amygdala coupling weakens, producing outsized emotional responses "
               "to small triggers.", 4, "established"),
            _f("brainstem", "Sleep-wake regulation destabilizes",
               "Adenosine accumulates and circadian signaling desynchronizes.", 3, "established"),
        ],
        "note": "Glymphatic clearance of metabolic waste largely occurs during sleep, which "
                "is why the effects feel physical as well as cognitive.",
    },
    {
        "keywords": ["exercise", "running", "cardio", "aerobic", "workout", "gym", "fitness"],
        "title": "Regular aerobic exercise",
        "summary": "The most robustly supported intervention for brain health in the "
                   "literature — and one of the few that increases hippocampal volume.",
        "findings": [
            _f("hippocampus", "Increased volume and neurogenesis", "Exercise raises BDNF, "
               "which supports survival of new neurons in the dentate gyrus. Randomized trials "
               "show measurable volume increases.", 5, "established", "strengthen"),
            _f("prefrontal_cortex", "Improved executive function and attention",
               "Increased perfusion and BDNF-driven synaptic plasticity.", 4, "established",
               "strengthen"),
            _f("cerebellum", "Refined motor timing and coordination",
               "Repeated complex movement drives procedural learning.", 3, "established",
               "strengthen"),
            _f("striatum", "Healthier dopamine tone", "Regular exercise supports dopamine "
               "receptor availability and stable reward signaling.", 3, "emerging", "strengthen"),
        ],
        "note": "Effects are dose-dependent and require sustained practice — a single "
                "session produces short-lived changes only.",
    },
    {
        "keywords": ["meditation", "mindfulness", "breathwork", "meditate"],
        "title": "Meditation and mindfulness practice",
        "summary": "Structural changes appear after sustained practice (weeks to months), "
                   "though early studies overstated effect sizes.",
        "findings": [
            _f("insula", "Thickened interoceptive cortex", "Repeated attention to bodily "
               "sensation drives use-dependent change in the insula.", 4, "established",
               "strengthen"),
            _f("amygdala", "Reduced reactivity and volume",
               "Longitudinal studies show decreased amygdala reactivity to stressors.",
               4, "established", "strengthen"),
            _f("prefrontal_cortex", "Stronger attention regulation",
               "Practice repeatedly exercises the return-attention-to-target loop.",
               3, "emerging", "strengthen"),
            _f("anterior_cingulate", "Improved conflict monitoring",
               "Noticing mind-wandering is itself an ACC function, trained by repetition.",
               3, "emerging", "strengthen"),
        ],
        "note": "Many early neuroimaging studies had small samples; effects are real but "
                "more modest than popular coverage suggests.",
    },
    {
        "keywords": ["instrument", "music", "piano", "guitar", "learning music", "practice music"],
        "title": "Learning a musical instrument",
        "summary": "One of the clearest demonstrations of structural plasticity — "
                   "measurable within weeks of daily practice.",
        "findings": [
            _f("motor_cortex", "Expanded representation for the trained hand",
               "Use-dependent cortical remapping enlarges the area devoted to trained digits.",
               5, "established", "strengthen"),
            _f("temporal_auditory", "Sharper pitch and timbre discrimination",
               "Auditory cortex tuning refines with repeated discrimination demands.",
               4, "established", "strengthen"),
            _f("cerebellum", "Precise timing and sequencing",
               "Procedural learning of motor sequences.", 4, "established", "strengthen"),
            _f("somatosensory_cortex", "Finer tactile discrimination",
               "Sensory maps for the fingers sharpen.", 3, "established", "strengthen"),
            _f("prefrontal_cortex", "Improved working memory and sustained focus",
               "Practice demands holding structure in mind over time.", 3, "emerging",
               "strengthen"),
        ],
        "note": "Effects are strongest with early and sustained training, but adult learners "
                "show real structural change too.",
    },
    {
        "keywords": ["language", "bilingual", "learning a language", "duolingo", "spanish",
                     "french", "new language"],
        "title": "Learning a new language",
        "summary": "Builds gray matter density in language and control regions; associated "
                   "with greater cognitive reserve later in life.",
        "findings": [
            _f("temporal_auditory", "Increased density in language areas",
               "Phonological and semantic processing demands drive local growth.",
               4, "established", "strengthen"),
            _f("prefrontal_cortex", "Stronger inhibitory control",
               "Bilinguals continually suppress the non-target language, exercising control "
               "circuits.", 3, "emerging", "strengthen"),
            _f("hippocampus", "Active vocabulary encoding",
               "Sustained new-word learning engages encoding machinery.", 3, "established",
               "strengthen"),
            _f("anterior_cingulate", "Improved conflict monitoring",
               "Language switching is a conflict-resolution task.", 2, "emerging", "strengthen"),
        ],
        "note": "The 'bilingual advantage' in executive function is contested — replication "
                "has been mixed, though structural findings hold up better.",
    },
    {
        "keywords": ["alcohol", "drinking", "drunk", "booze", "binge drinking"],
        "title": "Heavy alcohol use",
        "summary": "Affects memory, coordination, and reward circuitry; much is reversible "
                   "with sustained abstinence, but not all.",
        "findings": [
            _f("hippocampus", "Impaired encoding — blackouts and memory loss",
               "Alcohol blocks LTP in the hippocampus, preventing new memory formation "
               "while intoxicated.", 5, "established"),
            _f("cerebellum", "Degraded coordination and balance",
               "Cerebellar Purkinje cells are especially vulnerable to alcohol toxicity.",
               4, "established"),
            _f("prefrontal_cortex", "Reduced judgment and impulse control",
               "Prefrontal suppression is an acute effect and shows chronic changes with "
               "sustained heavy use.", 4, "established"),
            _f("striatum", "Reward system adapts toward dependence",
               "Repeated dopamine surges drive tolerance and habit circuitry.", 4, "established"),
        ],
        "note": "Substantial recovery is documented after sustained abstinence, though "
                "cerebellar damage can be lasting.",
    },
    {
        "keywords": ["lonely", "loneliness", "isolation", "social isolation", "no friends"],
        "title": "Chronic loneliness and social isolation",
        "summary": "Treated by the brain as a threat state — the effects overlap heavily "
                   "with chronic stress.",
        "findings": [
            _f("amygdala", "Heightened social-threat vigilance",
               "Isolation biases the brain toward detecting social threat, a self-reinforcing "
               "loop.", 4, "established"),
            _f("prefrontal_cortex", "Reduced executive function",
               "Chronic threat state consumes regulatory resources.", 3, "emerging"),
            _f("hippocampus", "Impaired memory and reduced neurogenesis",
               "Overlaps with the chronic-stress pathway.", 3, "emerging"),
            _f("insula", "Amplified distress signaling",
               "Social pain recruits partly overlapping circuitry with physical pain.",
               3, "emerging"),
        ],
        "note": "Loneliness is the subjective experience, not the objective number of "
                "contacts — the perception is what drives the biology.",
    },
    {
        "keywords": ["reading", "read books", "deep reading", "book"],
        "title": "Sustained deep reading",
        "summary": "Trains exactly the long-attention and integration circuits that "
                   "fragmented media under-exercises.",
        "findings": [
            _f("temporal_auditory", "Strengthened language and semantic networks",
               "Rich text repeatedly exercises semantic integration.", 4, "established",
               "strengthen"),
            _f("prefrontal_cortex", "Trained sustained attention",
               "Holding narrative structure across time is exactly the capacity that "
               "short-form content bypasses.", 4, "emerging", "strengthen"),
            _f("hippocampus", "Deeper encoding through elaboration",
               "Connecting new text to existing knowledge is elaborative encoding.",
               3, "established", "strengthen"),
            _f("parietal_cortex", "Integration across senses and perspective-taking",
               "Narrative comprehension engages spatial and perspective networks.",
               2, "emerging", "strengthen"),
        ],
        "note": "Useful as the direct counterweight to short-form consumption — it loads "
                "the same circuits that scrolling leaves idle.",
    },
]


def _norm(s):
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())


def lookup_curated(question):
    """Return the best curated entry for a question, or None."""
    q = _norm(question)
    best, best_score = None, 0
    for entry in CURATED:
        score = 0
        for kw in entry["keywords"]:
            if kw in q:
                # longer keyword matches are stronger evidence
                score = max(score, len(kw))
        if score > best_score:
            best, best_score = entry, score
    if best is None:
        return None
    result = dict(best)
    result["source"] = "curated"
    result.pop("keywords", None)
    return result


# ── Claude fallback (structured output) ──────────────────────────────────

_SYSTEM = """You explain what an activity, habit, or condition does to the human brain, \
for an educational 3D visualization. You are accurate and calibrated — this topic is full \
of pop-neuroscience myths and you do not repeat them.

Map the user's question onto these brain regions ONLY (use the exact id):
""" + "\n".join(f"- {r['id']}: {r['label']} — {r['blurb']}" for r in REGIONS) + """

Rules:
- Return 3 to 5 findings, ordered most affected first.
- severity is 1 (mildly engaged) to 5 (strongly affected).
- confidence must be honest:
    established  = well-replicated in humans
    emerging     = real but young/mixed evidence, often correlational
    hypothesized = plausible mechanism, not directly demonstrated
- direction is "impair" or "strengthen" — many activities strengthen regions.
- mechanism explains HOW, in one or two plain sentences a curious non-scientist \
understands. No jargon without explaining it.
- In `note`, state what the research actually supports and where it is weak or \
contested. If the premise of the question is a myth, say so plainly.
- If the question is not about the brain, set findings to an empty list and use \
`summary` to say so.
- This is educational, never diagnostic. Do not give medical advice."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "enum": REGION_IDS},
                    "effect": {"type": "string"},
                    "mechanism": {"type": "string"},
                    "severity": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    "confidence": {"type": "string", "enum": CONFIDENCE_LEVELS},
                    "direction": {"type": "string", "enum": ["impair", "strengthen"]},
                },
                "required": ["region", "effect", "mechanism", "severity",
                             "confidence", "direction"],
                "additionalProperties": False,
            },
        },
        "note": {"type": "string"},
    },
    "required": ["title", "summary", "findings", "note"],
    "additionalProperties": False,
}


def claude_available():
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def ask_claude(question):
    """Structured-output call. Raises on failure; caller decides the fallback."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=4000,
        system=[{"type": "text", "text": _SYSTEM,
                 "cache_control": {"type": "ephemeral"}}],
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
        messages=[{"role": "user", "content": question}],
    )

    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined to answer this question.")

    text = next(b.text for b in response.content if b.type == "text")
    data = json.loads(text)
    data["source"] = "claude"
    return data


def answer(question):
    """Hybrid: curated first, Claude for the long tail, honest message if neither."""
    curated = lookup_curated(question)
    if curated:
        return curated
    if claude_available():
        try:
            return ask_claude(question)
        except Exception as e:  # noqa: BLE001
            return {"source": "error", "title": "Couldn't answer that one",
                    "summary": f"The AI engine failed: {e}", "findings": [], "note": ""}
    return {
        "source": "unavailable",
        "title": "No entry for that yet",
        "summary": "This question isn't in the curated set, and the AI engine is not "
                   "configured. Set ANTHROPIC_API_KEY (and pip install anthropic) to "
                   "answer any question, or try one of the suggested topics.",
        "findings": [],
        "note": "Curated topics: " + ", ".join(e["title"] for e in CURATED),
    }


SUGGESTIONS = [
    "What happens to my brain when I watch excessive short-form content?",
    "What does chronic stress do to my brain?",
    "How does learning an instrument rewire my brain?",
    "What does sleep deprivation do to my brain?",
    "How does regular exercise change my brain?",
    "What happens when I meditate daily?",
]
