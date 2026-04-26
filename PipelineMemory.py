class PipelineMemory:
    def __init__(self):
        self.raw_df = None
        self.df = None
        self.target = None
        self.features = []
        self.models = {}
        self.metrics = {}
        self.report = {}