import yaml
import logging

logger = logging.getLogger(__name__)

class CurriculumManager:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.enabled = config.get('curriculum_enabled', False)
        if not self.enabled:
            self.stages = [{'opponent': 'self', 'timesteps': config.get('total_timesteps', 1e6)}] # Défaut: self-play
        else:
            self.stages = config.get('stages', [])
            if not self.stages:
                logger.warning("Curriculum learning activé mais aucune étape définie dans la config!")
                self.enabled = False 
                self.stages = [{'opponent': 'self', 'timesteps': config.get('total_timesteps', 1e6)}]

        self.current_stage_index = 0
        self.timesteps_in_current_stage = 0
        self.total_timesteps_across_stages = 0

        cumulative_ts = 0
        for stage in self.stages:
             cumulative_ts += stage['timesteps']
             stage['cumulative_timesteps'] = cumulative_ts


    def get_current_stage(self, total_timesteps: int) -> dict:
        """Détermine l'étape actuelle du curriculum en fonction des timesteps totaux."""
        if not self.enabled:
            return self.stages[0] 

        target_stage = self.stages[-1]
        for stage in self.stages:
            if total_timesteps < stage['cumulative_timesteps']:
                target_stage = stage
                break

        new_stage_index = self.stages.index(target_stage)
        if new_stage_index != self.current_stage_index:
             logger.info(f"Curriculum: Passage à l'étape {new_stage_index + 1}/{len(self.stages)}")
             logger.info(f"Nouvel adversaire: {target_stage.get('opponent', 'N/A')}")
             self.current_stage_index = new_stage_index

        return target_stage

    def get_opponent_type(self, total_timesteps: int) -> str:
        """Retourne le type d'adversaire pour l'étape actuelle."""
        stage = self.get_current_stage(total_timesteps)
        return stage.get('opponent', 'self') 

