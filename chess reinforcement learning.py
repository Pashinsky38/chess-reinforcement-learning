import chess
import chess.engine
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

class ChessNet(nn.Module):
    """Neural network to evaluate chess positions"""
    def __init__(self):
        super(ChessNet, self).__init__()
        # Input: 8x8x12 (piece positions) + other features
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)  # Value head - evaluates position
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        x = x.view(-1, 128 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class ChessRLAgent:
    def __init__(self, learning_rate=0.001, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessNet().to(self.device)
        self.target_model = ChessNet().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
    def board_to_tensor(self, board):
        """Convert chess board to neural network input tensor"""
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank, file = divmod(square, 8)
                piece_type = piece_map[piece.piece_type]
                channel = piece_type if piece.color == chess.WHITE else piece_type + 6
                tensor[channel, rank, file] = 1.0
        
        return torch.FloatTensor(tensor).unsqueeze(0).to(self.device)
    
    def get_legal_moves(self, board):
        """Get list of legal moves"""
        return list(board.legal_moves)
    
    def evaluate_move(self, board, move):
        """Evaluate a specific move"""
        board_copy = board.copy()
        board_copy.push(move)
        state = self.board_to_tensor(board_copy)
        
        with torch.no_grad():
            value = self.model(state).item()
        
        return value
    
    def select_move(self, board, training=True):
        """Select a move using epsilon-greedy policy"""
        legal_moves = self.get_legal_moves(board)
        
        if not legal_moves:
            return None
        
        # Exploration: random move
        if training and random.random() < self.epsilon:
            return random.choice(legal_moves)
        
        # Exploitation: best move according to model
        move_values = []
        for move in legal_moves:
            value = self.evaluate_move(board, move)
            # Negate value if it's opponent's turn after this move
            move_values.append(-value if board.turn == chess.WHITE else value)
        
        best_idx = np.argmax(move_values)
        return legal_moves[best_idx]
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            target = reward
            
            if not done:
                next_value = self.target_model(next_state).item()
                target = reward + self.gamma * next_value
            
            current_value = self.model(state)
            loss = nn.MSELoss()(current_value, torch.FloatTensor([[target]]).to(self.device))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        """Update target network"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def save_model(self, filepath):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model weights"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"Model loaded from {filepath}")
        else:
            print(f"No model found at {filepath}")

class ChessTrainer:
    def __init__(self, agent, engine_path=None):
        self.agent = agent
        self.engine = None
        self.engine_path = engine_path
        
        # Try to load Stockfish if path provided
        if engine_path and os.path.exists(engine_path):
            self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    
    def play_against_engine(self, engine_skill_level=1):
        """Play a game against chess engine"""
        board = chess.Board()
        states = []
        
        if self.engine:
            self.engine.configure({"Skill Level": engine_skill_level})
        
        while not board.is_game_over():
            # Agent's turn (White)
            if board.turn == chess.WHITE:
                state = self.agent.board_to_tensor(board)
                move = self.agent.select_move(board, training=True)
                
                if move is None:
                    break
                
                board.push(move)
                states.append((state, move, board.copy()))
                
            # Engine's turn (Black)
            else:
                if self.engine:
                    result = self.engine.play(board, chess.engine.Limit(time=0.1))
                    board.push(result.move)
                else:
                    # Random move if no engine
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        board.push(random.choice(legal_moves))
        
        # Determine reward
        result = board.result()
        if result == "1-0":  # Agent won
            reward = 1.0
        elif result == "0-1":  # Agent lost
            reward = -1.0
        else:  # Draw
            reward = 0.0
        
        # Store experiences
        for i, (state, move, board_state) in enumerate(states):
            next_state = self.agent.board_to_tensor(board_state)
            done = (i == len(states) - 1)
            self.agent.remember(state, move, reward, next_state, done)
        
        return result, reward
    
    def train(self, episodes=1000, update_target_every=10, save_every=100):
        """Train the agent"""
        print(f"Training on {self.agent.device}")
        print(f"Engine: {'Available' if self.engine else 'Not available (using random moves)'}")
        
        wins, losses, draws = 0, 0, 0
        
        for episode in range(episodes):
            # Gradually increase difficulty
            skill_level = min(1 + episode // 100, 20)
            
            result, reward = self.play_against_engine(engine_skill_level=skill_level)
            
            if result == "1-0":
                wins += 1
            elif result == "0-1":
                losses += 1
            else:
                draws += 1
            
            # Train on experiences
            self.agent.replay()
            
            # Update target network
            if episode % update_target_every == 0:
                self.agent.update_target_model()
            
            # Save model
            if episode % save_every == 0 and episode > 0:
                self.agent.save_model(f"chess_model_ep{episode}.pt")
            
            # Print progress
            if episode % 10 == 0:
                print(f"Episode {episode}/{episodes} | "
                      f"W: {wins} L: {losses} D: {draws} | "
                      f"Epsilon: {self.agent.epsilon:.3f} | "
                      f"Skill Level: {skill_level}")
                wins, losses, draws = 0, 0, 0
        
        if self.engine:
            self.engine.quit()

# Example usage
if __name__ == "__main__":
    # Create agent
    agent = ChessRLAgent(learning_rate=0.0001, gamma=0.95)
    
    # Optional: Load existing model
    # agent.load_model("chess_model_ep1000.pt")
    
    # Create trainer
    # Replace with your Stockfish path, or leave None for random opponent
    stockfish_path = None  # e.g., "/usr/local/bin/stockfish" or "stockfish.exe"
    trainer = ChessTrainer(agent, engine_path=stockfish_path)
    
    # Train
    trainer.train(episodes=1000)
    
    # Save final model
    agent.save_model("chess_model_final.pt")