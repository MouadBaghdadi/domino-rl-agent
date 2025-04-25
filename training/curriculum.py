from typing import Any, Dict, Optional
import yaml
import logging

logger = logging.getLogger(__name__)

class CurriculumManager:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.enabled = config.get('curriculum_enabled', False)
        default_total_timesteps = config.get('total_timesteps', 1e6) 

        if not self.enabled:
            self.stages = [{'opponent': 'self', 'timesteps': default_total_timesteps, 'cumulative_timesteps': default_total_timesteps}]
        else:
            self.stages = config.get('stages', [])
            if not self.stages:
                logger.warning("Curriculum learning activé mais aucune étape ('stages') définie dans la config!")
                self.enabled = False 
                self.stages = [{'opponent': 'self', 'timesteps': default_total_timesteps, 'cumulative_timesteps': default_total_timesteps}]
            else:
                cumulative_ts = 0
                valid_stages = []
                for stage in self.stages:
                     stage_ts = stage.get('timesteps')
                     if stage_ts is None or stage_ts <=0:
                          logger.warning(f"Étape de curriculum ignorée car 'timesteps' est manquant ou invalide: {stage}")
                          continue
                     cumulative_ts += stage_ts
                     stage['cumulative_timesteps'] = cumulative_ts
                     valid_stages.append(stage)
                self.stages = valid_stages 

                if cumulative_ts < default_total_timesteps:
                     logger.warning(f"Les étapes du curriculum ({cumulative_ts} ts) ne couvrent pas total_timesteps ({default_total_timesteps}). La dernière étape sera étendue.")

        self.current_stage_index = -1 
        # self.timesteps_in_current_stage = 0
        # self.total_timesteps_across_stages = 0


    def get_current_stage(self, total_timesteps: int) -> dict:
        """
        Détermine l'étape actuelle du curriculum en fonction des timesteps totaux.

        Args:
            total_timesteps: Nombre total de pas d'environnement effectués depuis le début.

        Returns:
            Un dictionnaire représentant la configuration de l'étape actuelle.
        """
        if not self.stages:
             return {'opponent': 'self'} 

        target_stage = self.stages[-1]
        target_stage_index = len(self.stages) - 1

        for i, stage in enumerate(self.stages):
            if total_timesteps < stage['cumulative_timesteps']:
                target_stage = stage
                target_stage_index = i
                break

        if target_stage_index != self.current_stage_index:
             opponent = target_stage.get('opponent', 'N/A') if isinstance(target_stage, dict) else 'N/A'
             logger.info(f"Curriculum: Entrée/Passage à l'étape {target_stage_index + 1}/{len(self.stages)} @ {total_timesteps} timesteps.")
             logger.info(f"  -> Configuration étape: {target_stage}") 
             self.current_stage_index = target_stage_index

        # return target_stage.copy()
        return target_stage

    def get_parameter(self, param_name: str, total_timesteps: int, default_value: Any = None) -> Any:
         """
         Récupère une valeur de paramètre spécifique pour l'étape actuelle du curriculum.

         Args:
             param_name: Le nom du paramètre à récupérer (ex: 'opponent', 'learning_rate').
             total_timesteps: Le nombre total de timesteps actuel.
             default_value: La valeur à retourner si le paramètre n'est pas défini
                            pour l'étape actuelle.

         Returns:
             La valeur du paramètre pour l'étape actuelle ou la valeur par défaut.
         """
         stage = self.get_current_stage(total_timesteps)
         if isinstance(stage, dict):
              return stage.get(param_name, default_value)
         else: 
              logger.error("Erreur interne: 'stage' n'est pas un dictionnaire dans get_parameter.")
              return default_value


    def get_opponent_type(self, total_timesteps: int) -> str:
        """Retourne le type d'adversaire pour l'étape actuelle."""
        return self.get_parameter('opponent', total_timesteps, default_value='self')

    def get_current_learning_rate(self, total_timesteps: int, default_lr: float) -> float:
        """Retourne le learning rate pour l'étape actuelle, ou le défaut."""
        return self.get_parameter('learning_rate', total_timesteps, default_value=default_lr)

    def get_current_entropy_coeff(self, total_timesteps: int, default_entropy: float) -> float:
        """Retourne le coefficient d'entropie pour l'étape actuelle, ou le défaut."""
        return self.get_parameter('entropy_coeff', total_timesteps, default_value=default_entropy)

    def get_current_reward_shaping_config(self, total_timesteps: int) -> Optional[Dict[str, Any]]:
         """Retourne la config de reward shaping pour l'étape actuelle, ou None."""
         return self.get_parameter('reward_shaping', total_timesteps, default_value=None)