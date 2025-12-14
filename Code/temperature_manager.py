"""
Temperature management module for adaptive response strategies.
This module provides functionality for dynamically adjusting temperature
based on conversation progress and success metrics.
"""

class TemperatureManager:
    """
    Manages temperature settings for LLM interactions based on success metrics
    and adaptive strategies.
    """
    
    def __init__(self, initial_temperature=0.7, min_temp=0.1, max_temp=1.0, success_threshold=0.5):
        """
        Initialize the temperature manager.
        
        Args:
            initial_temperature (float): Starting temperature (default: 0.7)
            min_temp (float): Minimum temperature value (default: 0.1)
            max_temp (float): Maximum temperature value (default: 1.0)
            success_threshold (float): Threshold for considering an attempt successful (default: 0.5)
        """
        self.current_temperature = initial_temperature
        self.initial_temperature = initial_temperature
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.success_threshold = success_threshold
        self.success_history = []
        self.temperature_history = [initial_temperature]
        
        # Strategy coordination state
        self.last_strategy_used = None
        self.strategy_change_count = 0
        self.consecutive_strategy_uses = 0
    
    def adjust_temperature(self, success_indicator, strategy="adaptive"):
        """
        Adjust temperature based on success of previous interactions.
        Only adjusts for unsuccessful attempts (below success threshold).
        
        Args:
            success_indicator (float): Value between 0 and 1 indicating success of last interaction
            strategy (str): Strategy to use for temperature adjustment 
                           ("adaptive", "oscillating", "progressive", or "reset")
        
        Returns:
            float: The new temperature value
        """
        self.success_history.append(success_indicator)
        
        # Strategy coordination tracking
        if strategy != self.last_strategy_used:
            self.strategy_change_count += 1
            self.consecutive_strategy_uses = 1
        else:
            self.consecutive_strategy_uses += 1
        self.last_strategy_used = strategy
        
        # Only adjust temperature for unsuccessful attempts
        if success_indicator >= self.success_threshold:
            # Success! No temperature adjustment needed
            # Keep current temperature for consistency
            pass
        else:
            # Failed attempt - adjust temperature to improve next try
            # Store pre-adjustment temperature for conflict detection
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
                # Default to adaptive
                self._adjust_adaptive()
            
            # Prevent extreme changes when strategies conflict
            temp_change = abs(self.current_temperature - pre_adjustment_temp)
            max_single_change = (self.max_temp - self.min_temp) * 0.3  # Max 30% of range per step
            
            if temp_change > max_single_change:
                # Limit the change to prevent strategy conflicts
                change_direction = 1 if self.current_temperature > pre_adjustment_temp else -1
                self.current_temperature = pre_adjustment_temp + (change_direction * max_single_change)
            
        # Ensure temperature stays within bounds
        self.current_temperature = max(self.min_temp, min(self.max_temp, self.current_temperature))
        self.temperature_history.append(self.current_temperature)
        
        return self.current_temperature
    
    def _adjust_adaptive(self):
        """Adaptive temperature adjustment based on recent success (only for failed attempts)"""
        # Get the last 3 success indicators, or fewer if history is shorter
        # Only consider unsuccessful attempts (below threshold) for temperature adjustment
        recent_failed = [score for score in self.success_history[-3:] if score < self.success_threshold]
        
        if not recent_failed:
            # No recent failures to learn from, keep current temperature
            return
            
        avg_failed_score = sum(recent_failed) / len(recent_failed)
        
        # Adjust based on how close failed attempts are getting to success threshold
        score_gap = self.success_threshold - avg_failed_score
        
        if score_gap > (self.success_threshold * 0.5):  # Far from success (e.g., 0.3+ gap)
            # Increase temperature significantly for more creativity
            self.current_temperature += 0.1
        elif score_gap > (self.success_threshold * 0.25):  # Moderate distance (e.g., 0.15+ gap)
            # Small increase to add some variability
            self.current_temperature += 0.05
        else:  # Close to success threshold
            # Very small adjustment or none - we're almost there
            self.current_temperature += 0.02
    
    def _adjust_oscillating(self):
        """Smart oscillation with controlled exploration to escape local minima"""
        recent_failed = [score for score in self.success_history[-4:] if score < self.success_threshold]
        
        if len(recent_failed) < 3:
            # Not enough data for pattern detection, use adaptive approach
            self._adjust_adaptive()
            return
        
        # Detect if we're stuck in a local minimum (low variance in scores)
        score_variance = max(recent_failed) - min(recent_failed)
        avg_recent_score = sum(recent_failed) / len(recent_failed)
        
        # Check if we're genuinely stuck
        is_stuck = (
            score_variance < 0.08 and  # Low variance in scores
            avg_recent_score < (self.success_threshold * 0.7) and  # Not close to success
            len(set(round(score, 2) for score in recent_failed)) <= 2  # Clustering around similar values
        )
        
        if is_stuck:
            # Controlled oscillation to escape local minimum
            temp_range = self.max_temp - self.min_temp
            
            # Check recent temperature history to avoid immediate repetition
            if len(self.temperature_history) >= 2:
                last_two_temps = self.temperature_history[-2:]
                temp_direction = last_two_temps[-1] - last_two_temps[-2]
                
                if abs(temp_direction) < 0.05:  # Temperature hasn't changed much
                    # Make a deliberate exploration move
                    if self.current_temperature < self.initial_temperature:
                        self.current_temperature = min(self.max_temp, self.current_temperature + temp_range * 0.25)
                    else:
                        self.current_temperature = max(self.min_temp, self.current_temperature - temp_range * 0.25)
                else:
                    # Continue the direction but with moderation
                    self.current_temperature += temp_direction * 0.5
            else:
                # First oscillation attempt
                if avg_recent_score < (self.success_threshold * 0.5):
                    self.current_temperature += 0.15  # Try higher creativity
                else:
                    self.current_temperature -= 0.1   # Try more focus
        else:
            # Not stuck - use adaptive approach to avoid unnecessary oscillation
            self._adjust_adaptive()
    
    def _adjust_progressive(self):
        """Progressive temperature changes based on failed attempts trajectory"""
        # Only consider unsuccessful attempts for trajectory analysis
        recent_failed = [score for score in self.success_history[-5:] if score < self.success_threshold]
        
        if len(recent_failed) >= 3:
            # Calculate trajectory of failed attempts (are they getting closer to success?)
            trajectory = sum(y - x for x, y in zip(recent_failed[-3:-1], recent_failed[-2:]))
            
            # If failed attempts are improving (getting closer to threshold), keep current approach
            if trajectory > 0:
                self.current_temperature += 0.02  # Small increase to push over the line
            # If failed attempts are getting worse, increase temperature more
            elif trajectory < 0:
                self.current_temperature += 0.08
            # If stable failure, adjust based on distance from threshold
            else:
                last_failed_score = recent_failed[-1]
                score_gap = self.success_threshold - last_failed_score
                if score_gap > (self.success_threshold * 0.4):  # Far from success
                    self.current_temperature += 0.06
                else:  # Close to success
                    self.current_temperature += 0.03
        else:
            # Not enough failed attempts, use adaptive approach
            self._adjust_adaptive()
    
    def _adjust_reset(self):
        """Smart reset based on sustained lack of progress over time"""
        if len(self.success_history) < 5:
            # Not enough history, use adaptive
            self._adjust_adaptive()
            return
        
        # Analyze progress over different time windows
        recent_scores = [score for score in self.success_history[-5:] if score < self.success_threshold]
        
        if not recent_scores:
            # No recent failures to analyze
            return
        
        # Check for sustained lack of progress
        earliest_score = recent_scores[0]
        latest_score = recent_scores[-1]
        progress = latest_score - earliest_score
        
        # Check if we're making negative or no progress AND scores are poor
        avg_recent = sum(recent_scores) / len(recent_scores)
        is_regression = progress < -0.05 and avg_recent < (self.success_threshold * 0.5)  # Getting worse + poor scores
        is_stagnant = abs(progress) < 0.03 and avg_recent < (self.success_threshold * 0.5)  # No progress + poor scores
        
        # Also check temperature effectiveness
        if len(self.temperature_history) >= 5:
            temp_variance = max(self.temperature_history[-5:]) - min(self.temperature_history[-5:])
            tried_wide_range = temp_variance > (self.max_temp - self.min_temp) * 0.4
        else:
            tried_wide_range = False
        
        if (is_regression or is_stagnant) and tried_wide_range:
            # We've tried various temperatures but getting worse or stagnant
            # Reset to a strategically better value, not just initial
            
            score_gap = self.success_threshold - avg_recent
            
            if score_gap > (self.success_threshold * 0.5):  # Very far from success
                # Reset to higher temperature for more exploration
                target_temp = self.initial_temperature + (self.max_temp - self.initial_temperature) * 0.4
            else:  # Moderately far from success
                # Reset to moderate temperature
                target_temp = self.initial_temperature + (self.max_temp - self.initial_temperature) * 0.2
                
            # Clamp to bounds
            self.current_temperature = max(self.min_temp, min(self.max_temp, target_temp))
        else:
            # Normal situation - use adaptive
            self._adjust_adaptive()
    
    def get_current_temperature(self):
        """Get the current temperature value"""
        return self.current_temperature
    
    def reset(self):
        """Reset temperature to initial value"""
        self.current_temperature = self.initial_temperature
        return self.current_temperature
    
    def get_temperature_history(self):
        """Get the temperature history list"""
        return self.temperature_history
    
    def get_success_history(self):
        """Get the success history list"""
        return self.success_history
    
    def get_strategy_stats(self):
        """Get statistics about strategy usage and effectiveness"""
        return {
            "last_strategy_used": self.last_strategy_used,
            "strategy_changes": self.strategy_change_count,
            "consecutive_uses": self.consecutive_strategy_uses,
            "temperature_range_explored": max(self.temperature_history) - min(self.temperature_history) if self.temperature_history else 0,
            "current_temperature": self.current_temperature,
            "recent_failed_scores": [score for score in self.success_history[-5:] if score < self.success_threshold]
        }
    
    def _detect_strategy_conflicts(self, strategy):
        """Detect potential conflicts between strategies"""
        if len(self.temperature_history) < 3:
            return False
        
        # Check for rapid temperature oscillations that might indicate strategy conflicts
        recent_temps = self.temperature_history[-3:]
        temp_changes = [abs(recent_temps[i] - recent_temps[i-1]) for i in range(1, len(recent_temps))]
        
        # Large consecutive changes might indicate conflicting strategies
        if temp_changes:
            avg_change = sum(temp_changes) / len(temp_changes)
            large_change_threshold = (self.max_temp - self.min_temp) * 0.15
            return avg_change > large_change_threshold and self.strategy_change_count > 2
        
        return False
    
    def reset_coordination_state(self):
        """Reset strategy coordination tracking (useful for new conversations)"""
        self.last_strategy_used = None
        self.strategy_change_count = 0
        self.consecutive_strategy_uses = 0

    
    def recommend_strategy(self, turn_number=1, base_strategy="adaptive"):
        """
        Intelligently recommend the best temperature adjustment strategy based on conversation context.
        
        Args:
            turn_number (int): Current turn in the conversation
            base_strategy (str): Base strategy from configuration
        
        Returns:
            str: Recommended strategy ("adaptive", "oscillating", "progressive", "reset")
        """
        recent_failed = [score for score in self.success_history[-5:] if score < self.success_threshold]
        
        # Not enough data - use base strategy
        if len(recent_failed) < 2:
            return base_strategy
        
        # Analyze conversation patterns
        latest_scores = recent_failed[-3:] if len(recent_failed) >= 3 else recent_failed
        avg_recent_score = sum(latest_scores) / len(latest_scores)
        score_variance = max(latest_scores) - min(latest_scores) if len(latest_scores) > 1 else 0
        
        # Calculate progress trend
        if len(recent_failed) >= 3:
            early_avg = sum(recent_failed[:2]) / 2
            late_avg = sum(recent_failed[-2:]) / 2
            progress_trend = late_avg - early_avg
        else:
            progress_trend = 0
        
        # Check temperature effectiveness
        temp_variance = 0
        if len(self.temperature_history) >= 3:
            recent_temps = self.temperature_history[-3:]
            temp_variance = max(recent_temps) - min(recent_temps)
        
        # Decision logic based on conversation analysis
        
        # 1. RESET: When sustained poor performance despite temperature exploration
        if (len(recent_failed) >= 4 and 
            avg_recent_score < (self.success_threshold * 0.4) and
            progress_trend < -0.02 and  # Getting worse
            temp_variance > (self.max_temp - self.min_temp) * 0.3):  # Tried various temps
            return "reset"
        
        # 2. OSCILLATING: When stuck in local minimum (low variance, poor scores)
        if (score_variance < 0.06 and 
            avg_recent_score < (self.success_threshold * 0.7) and
            len(set(round(score, 2) for score in latest_scores)) <= 2 and  # Clustering
            self.consecutive_strategy_uses >= 2):  # Same strategy not working
            return "oscillating"
        
        # 3. PROGRESSIVE: When showing improvement trend or in final turns
        if (progress_trend > 0.03 or  # Improving
            turn_number >= 4 or  # Final push
            (avg_recent_score > (self.success_threshold * 0.8) and turn_number >= 3)):  # Close to success
            return "progressive"
        
        # 4. ADAPTIVE: Default for exploration and general adjustment
        return "adaptive"
    
    def analyze_conversation_state(self):
        """
        Analyze the current conversation state for debugging and optimization.
        
        Returns:
            dict: Analysis of conversation patterns and recommendations
        """
        recent_failed = [score for score in self.success_history[-5:] if score < self.success_threshold]
        
        if not recent_failed:
            return {"state": "no_data", "recommendation": "adaptive"}
        
        # Calculate metrics
        avg_score = sum(recent_failed) / len(recent_failed)
        score_variance = max(recent_failed) - min(recent_failed) if len(recent_failed) > 1 else 0
        
        # Progress analysis
        progress_trend = 0
        if len(recent_failed) >= 3:
            early_avg = sum(recent_failed[:2]) / 2
            late_avg = sum(recent_failed[-2:]) / 2
            progress_trend = late_avg - early_avg
        
        # Temperature analysis
        temp_variance = 0
        if len(self.temperature_history) >= 3:
            recent_temps = self.temperature_history[-3:]
            temp_variance = max(recent_temps) - min(recent_temps)
        
        # Determine state
        if avg_score < (self.success_threshold * 0.3):
            state = "very_poor"
        elif avg_score < (self.success_threshold * 0.6):
            state = "poor"
        elif avg_score < (self.success_threshold * 0.9):
            state = "close"
        else:
            state = "very_close"
        
        # Pattern detection
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


