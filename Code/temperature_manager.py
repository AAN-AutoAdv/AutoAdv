class TemperatureManager:
    def __init__(self, initial_temperature=0.7, min_temp=0.1, max_temp=1.0, success_threshold=0.5):
        self.current_temperature = initial_temperature
        self.initial_temperature = initial_temperature
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.success_threshold = success_threshold
        self.success_history = []
        self.temperature_history = [initial_temperature]

        self.last_strategy_used = None
        self.strategy_change_count = 0
        self.consecutive_strategy_uses = 0
    
    def adjust_temperature(self, success_indicator, strategy="adaptive"):
        self.success_history.append(success_indicator)

        if strategy != self.last_strategy_used:
            self.strategy_change_count += 1
            self.consecutive_strategy_uses = 1
        else:
            self.consecutive_strategy_uses += 1
        self.last_strategy_used = strategy

        if success_indicator >= self.success_threshold:
            pass
        else:
            pre_adjustment_temp = self.current_temperature
            
            if strategy == "adaptive":
                self._adjust_adaptive()
            elif strategy == "oscillating":
                self._adjust_oscillating()
            elif strategy == "progressive":
                self._adjust_progressive()
            elif strategy == "reset":
                self._adjust_reset()
            else:
                self._adjust_adaptive()

            temp_change = abs(self.current_temperature - pre_adjustment_temp)
            max_single_change = (self.max_temp - self.min_temp) * 0.3

            if temp_change > max_single_change:
                change_direction = 1 if self.current_temperature > pre_adjustment_temp else -1
                self.current_temperature = pre_adjustment_temp + (change_direction * max_single_change)

        self.current_temperature = max(self.min_temp, min(self.max_temp, self.current_temperature))
        self.temperature_history.append(self.current_temperature)
        
        return self.current_temperature
    
    def _adjust_adaptive(self):
        recent_failed = [score for score in self.success_history[-3:] if score < self.success_threshold]

        if not recent_failed:
            return

        avg_failed_score = sum(recent_failed) / len(recent_failed)
        score_gap = self.success_threshold - avg_failed_score

        if score_gap > (self.success_threshold * 0.5):
            self.current_temperature += 0.1
        elif score_gap > (self.success_threshold * 0.25):
            self.current_temperature += 0.05
        else:
            self.current_temperature += 0.02
    
    def _adjust_oscillating(self):
        recent_failed = [score for score in self.success_history[-4:] if score < self.success_threshold]

        if len(recent_failed) < 3:
            self._adjust_adaptive()
            return

        score_variance = max(recent_failed) - min(recent_failed)
        avg_recent_score = sum(recent_failed) / len(recent_failed)

        is_stuck = (
            score_variance < 0.08 and
            avg_recent_score < (self.success_threshold * 0.7) and
            len(set(round(score, 2) for score in recent_failed)) <= 2
        )

        if is_stuck:
            temp_range = self.max_temp - self.min_temp

            if len(self.temperature_history) >= 2:
                last_two_temps = self.temperature_history[-2:]
                temp_direction = last_two_temps[-1] - last_two_temps[-2]

                if abs(temp_direction) < 0.05:
                    if self.current_temperature < self.initial_temperature:
                        self.current_temperature = min(self.max_temp, self.current_temperature + temp_range * 0.25)
                    else:
                        self.current_temperature = max(self.min_temp, self.current_temperature - temp_range * 0.25)
                else:
                    self.current_temperature += temp_direction * 0.5
            else:
                if avg_recent_score < (self.success_threshold * 0.5):
                    self.current_temperature += 0.15
                else:
                    self.current_temperature -= 0.1
        else:
            self._adjust_adaptive()
    
    def _adjust_progressive(self):
        recent_failed = [score for score in self.success_history[-5:] if score < self.success_threshold]

        if len(recent_failed) >= 3:
            trajectory = sum(y - x for x, y in zip(recent_failed[-3:-1], recent_failed[-2:]))

            if trajectory > 0:
                self.current_temperature += 0.02
            elif trajectory < 0:
                self.current_temperature += 0.08
            else:
                last_failed_score = recent_failed[-1]
                score_gap = self.success_threshold - last_failed_score
                if score_gap > (self.success_threshold * 0.4):
                    self.current_temperature += 0.06
                else:
                    self.current_temperature += 0.03
        else:
            self._adjust_adaptive()
    
    def _adjust_reset(self):
        if len(self.success_history) < 5:
            self._adjust_adaptive()
            return

        recent_scores = [score for score in self.success_history[-5:] if score < self.success_threshold]

        if not recent_scores:
            return

        earliest_score = recent_scores[0]
        latest_score = recent_scores[-1]
        progress = latest_score - earliest_score

        avg_recent = sum(recent_scores) / len(recent_scores)
        is_regression = progress < -0.05 and avg_recent < (self.success_threshold * 0.5)
        is_stagnant = abs(progress) < 0.03 and avg_recent < (self.success_threshold * 0.5)
        if len(self.temperature_history) >= 5:
            temp_variance = max(self.temperature_history[-5:]) - min(self.temperature_history[-5:])
            tried_wide_range = temp_variance > (self.max_temp - self.min_temp) * 0.4
        else:
            tried_wide_range = False
        
        if (is_regression or is_stagnant) and tried_wide_range:
            score_gap = self.success_threshold - avg_recent

            if score_gap > (self.success_threshold * 0.5):
                target_temp = self.initial_temperature + (self.max_temp - self.initial_temperature) * 0.4
            else:
                target_temp = self.initial_temperature + (self.max_temp - self.initial_temperature) * 0.2

            self.current_temperature = max(self.min_temp, min(self.max_temp, target_temp))
        else:
            self._adjust_adaptive()
    
    def get_current_temperature(self):
        return self.current_temperature
    
    def reset(self):
        self.current_temperature = self.initial_temperature
        return self.current_temperature
    
    def get_temperature_history(self):
        return self.temperature_history
    
    def get_success_history(self):
        return self.success_history
    
    def get_strategy_stats(self):
        return {
            "last_strategy_used": self.last_strategy_used,
            "strategy_changes": self.strategy_change_count,
            "consecutive_uses": self.consecutive_strategy_uses,
            "temperature_range_explored": max(self.temperature_history) - min(self.temperature_history) if self.temperature_history else 0,
            "current_temperature": self.current_temperature,
            "recent_failed_scores": [score for score in self.success_history[-5:] if score < self.success_threshold]
        }
    
    def _detect_strategy_conflicts(self, strategy):
        if len(self.temperature_history) < 3:
            return False

        recent_temps = self.temperature_history[-3:]
        temp_changes = [abs(recent_temps[i] - recent_temps[i-1]) for i in range(1, len(recent_temps))]

        if temp_changes:
            avg_change = sum(temp_changes) / len(temp_changes)
            large_change_threshold = (self.max_temp - self.min_temp) * 0.15
            return avg_change > large_change_threshold and self.strategy_change_count > 2
        
        return False
    
    def reset_coordination_state(self):
        self.last_strategy_used = None
        self.strategy_change_count = 0
        self.consecutive_strategy_uses = 0

    
    def recommend_strategy(self, turn_number=1, base_strategy="adaptive"):
        recent_failed = [score for score in self.success_history[-5:] if score < self.success_threshold]

        if len(recent_failed) < 2:
            return base_strategy

        latest_scores = recent_failed[-3:] if len(recent_failed) >= 3 else recent_failed
        avg_recent_score = sum(latest_scores) / len(latest_scores)
        score_variance = max(latest_scores) - min(latest_scores) if len(latest_scores) > 1 else 0

        if len(recent_failed) >= 3:
            early_avg = sum(recent_failed[:2]) / 2
            late_avg = sum(recent_failed[-2:]) / 2
            progress_trend = late_avg - early_avg
        else:
            progress_trend = 0

        temp_variance = 0
        if len(self.temperature_history) >= 3:
            recent_temps = self.temperature_history[-3:]
            temp_variance = max(recent_temps) - min(recent_temps)

        if (len(recent_failed) >= 4 and
            avg_recent_score < (self.success_threshold * 0.4) and
            progress_trend < -0.02 and
            temp_variance > (self.max_temp - self.min_temp) * 0.3):
            return "reset"

        if (score_variance < 0.06 and
            avg_recent_score < (self.success_threshold * 0.7) and
            len(set(round(score, 2) for score in latest_scores)) <= 2 and
            self.consecutive_strategy_uses >= 2):
            return "oscillating"

        if (progress_trend > 0.03 or
            turn_number >= 4 or
            (avg_recent_score > (self.success_threshold * 0.8) and turn_number >= 3)):
            return "progressive"

        return "adaptive"
    
    def analyze_conversation_state(self):
        recent_failed = [score for score in self.success_history[-5:] if score < self.success_threshold]
        
        if not recent_failed:
            return {"state": "no_data", "recommendation": "adaptive"}

        avg_score = sum(recent_failed) / len(recent_failed)
        score_variance = max(recent_failed) - min(recent_failed) if len(recent_failed) > 1 else 0

        progress_trend = 0
        if len(recent_failed) >= 3:
            early_avg = sum(recent_failed[:2]) / 2
            late_avg = sum(recent_failed[-2:]) / 2
            progress_trend = late_avg - early_avg

        temp_variance = 0
        if len(self.temperature_history) >= 3:
            recent_temps = self.temperature_history[-3:]
            temp_variance = max(recent_temps) - min(recent_temps)

        if avg_score < (self.success_threshold * 0.3):
            state = "very_poor"
        elif avg_score < (self.success_threshold * 0.6):
            state = "poor"
        elif avg_score < (self.success_threshold * 0.9):
            state = "close"
        else:
            state = "very_close"

        patterns = []
        if score_variance < 0.05:
            patterns.append("stuck")
        if progress_trend > 0.05:
            patterns.append("improving")
        elif progress_trend < -0.05:
            patterns.append("declining")
        if temp_variance > (self.max_temp - self.min_temp) * 0.4:
            patterns.append("explored_temps")
        
        return {
            "state": state,
            "avg_score": avg_score,
            "score_variance": score_variance,
            "progress_trend": progress_trend,
            "temp_variance": temp_variance,
            "patterns": patterns,
            "recent_failed_count": len(recent_failed)
        }


