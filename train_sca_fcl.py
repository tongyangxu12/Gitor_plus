import pandas as pd
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from nodevectors import ProNE
from datetime import datetime
import time
import numpy as np
import warnings


warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('Device: ', device)

class CodeCloneDetector_3(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2 * input_dim, 4 * input_dim),
            nn.ReLU(),
            nn.Linear(4 * input_dim, 2 * input_dim),
            nn.ReLU(),
            nn.Linear(2 * input_dim, 2),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class CodeCloneDetector_2(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2 * input_dim, 4 * input_dim),
            nn.ReLU(),
            nn.Linear(4 * input_dim, 2),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class CodeCloneDetector_1(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2 * input_dim, 2),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class ClonePairDataset(Dataset):
    def __init__(self, file_path, prone_model, dim):

        self.data = pd.read_csv(file_path)
        self.prone_model = prone_model
        self.prone_model.embedding_size = dim

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        f1 = self.data.iloc[idx]['f1']
        f2 = self.data.iloc[idx]['f2']
        type_label = self.data.iloc[idx]['type']

        try:
            vector1 = self.prone_model.predict(f'{f1}.java')  # Gets the vector of the first file
            vector2 = self.prone_model.predict(f'{f2}.java')  # Gets the vector of the second file
            vector1 = torch.tensor(vector1, dtype=torch.float32)
            vector2 = torch.tensor(vector2, dtype=torch.float32)
        except KeyError:

            vector1 = torch.zeros(self.prone_model.embedding_size, dtype=torch.float32)
            vector2 = torch.zeros(self.prone_model.embedding_size, dtype=torch.float32)

        concatenated_vector = torch.cat((vector1, vector2), dim=0)

        return concatenated_vector, type_label


def train_1(input_dim, model, i):
    learning_rate = 0.0001
    batch_size = 64
    epochs = 10
    total_time = 0

    # model = CodeCloneDetector(input_dim)
    model = CodeCloneDetector_1(input_dim).to(device)  # GPU

    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    w1 = ProNE.load(f'./metrics_keyword/dim_{input_dim}/embedding.zip')
    dataset = ClonePairDataset('../data/split_data/all.csv', w1, input_dim)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print('Total of dataset: ', len(dataset))
    print('Total of train_loader: ', len(train_loader))
    total_step = len(train_loader)

    model.train()
    size = len(dataset)
    start_time = time.time()
    for epoch in range(epochs):
        print(f"Start {epoch + 1} time is:", datetime.now().strftime("%H:%M"))
        begin_time = time.time()
        train_correct_total = 0
        loss_history = []
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            # 1.Empty gradient
            optimizer.zero_grad()
            # 2.Forward propagation
            pred = model(X)
            # print('---')
            # print(batch)
            # print(pred)
            # print(y)
            # 3.Calculate loss
            loss = loss_fn(pred, y)

            # Get prediction results
            # pred is the raw output that needs to be converted to probabilities and then indexed to the maximum value
            predicted_labels = torch.argmax(pred, dim=1)  # Gets the category with the greatest probability

            # Calculate the number of correct predictions in this batch
            correct = (predicted_labels == y).sum().item()
            print(f'correct: {correct}')
            train_correct_total += correct
            total = y.size(0)  # Add up the number of batches of samples

            # Accuracy rate for printing each batch (optional)
            batch_accuracy = (predicted_labels == y).sum().item() / y.size(0)
            print(f'Batch {batch}, Accuracy: {batch_accuracy:.4f}')

            # 4.Back propagation
            loss.backward()
            # 5.Update parameter
            optimizer.step()

            # 7. Print training information (optional)
            if (batch + 1) % 100 == 0:
                loss_history.append(loss.item())
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch + 1, epochs, batch + 1,
                                                                                           total_step,
                                                                                           np.mean(loss_history),
                                                                                           (correct / total) * 100))
        # Trained all batches
        print(f"Total correct predictions: {train_correct_total}")

        print('Training Loss: {:.4f}, Training Accuracy: {:.4f}'.format(np.mean(loss_history), (
                train_correct_total / len(dataset)) * 100))
        print(f'Training Time: {time.time() - begin_time}s')
        total_time += time.time() - begin_time
        with open(f'./output_sca_{i}/1_layers/dim_{input_dim}/' + 'train_result.txt', 'a+') as ff:
            ff.write(f'\nModel structure: {model}\n')
            ff.write(f'Dim: {input_dim}\n')
            ff.write(f'Training Time: {time.time() - begin_time}s\n')
            ff.write('Training Loss: {:.4f}, Training Accuracy: {:.4f}\n\n'.format(np.mean(loss_history), (
                    train_correct_total / len(dataset)) * 100))
            ff.close()

        # save the model
        torch.save(model.state_dict(), f'./output_sca_{i}/1_layers/dim_{input_dim}/' + 'model_' + str(epoch + 1) + '.pth')
        print(f"current epoch: {epoch + 1}, time: {time.time() - begin_time}s")

    with open(f'./output_sca_{i}/1_layers/dim_{input_dim}/' + 'train_result.txt', 'a+') as ff:
        ff.write(f'\n\ntotal time: {total_time}')
        ff.close()
    print(f"total time: {time.time() - start_time}s")

def train_2(input_dim, model, i):
    learning_rate = 0.0001
    batch_size = 64
    epochs = 10
    total_time = 0

    # model = CodeCloneDetector(input_dim)
    model = CodeCloneDetector_2(input_dim).to(device)  # GPU

    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    w1 = ProNE.load(f'./metrics_keyword/dim_{input_dim}/embedding.zip')
    dataset = ClonePairDataset('../data/split_data/all.csv', w1, input_dim)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print('Total of dataset: ', len(dataset))
    print('Total of train_loader: ', len(train_loader))
    total_step = len(train_loader)

    model.train()
    size = len(dataset)
    start_time = time.time()
    for epoch in range(epochs):
        print(f"Start {epoch + 1} time is:", datetime.now().strftime("%H:%M"))
        begin_time = time.time()
        train_correct_total = 0
        loss_history = []
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()

            pred = model(X)
            # print('---')
            # print(batch)
            # print(pred)
            # print(y)

            loss = loss_fn(pred, y)


            predicted_labels = torch.argmax(pred, dim=1)


            correct = (predicted_labels == y).sum().item()
            print(f'correct: {correct}')
            train_correct_total += correct
            total = y.size(0)


            batch_accuracy = (predicted_labels == y).sum().item() / y.size(0)
            print(f'Batch {batch}, Accuracy: {batch_accuracy:.4f}')


            loss.backward()

            optimizer.step()


            if (batch + 1) % 100 == 0:
                loss_history.append(loss.item())
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch + 1, epochs, batch + 1,
                                                                                           total_step,
                                                                                           np.mean(loss_history),
                                                                                           (correct / total) * 100))

        print(f"Total correct predictions: {train_correct_total}")

        print('Training Loss: {:.4f}, Training Accuracy: {:.4f}'.format(np.mean(loss_history), (
                train_correct_total / len(dataset)) * 100))
        print(f'Training Time: {time.time() - begin_time}s')
        total_time += time.time() - begin_time
        with open(f'./output_sca_{i}/2_layers/dim_{input_dim}/' + 'train_result.txt', 'a+') as ff:
            ff.write(f'\nModel structure: {model}\n')
            ff.write(f'Dim: {input_dim}\n')
            ff.write(f'Training Time: {time.time() - begin_time}s\n')
            ff.write('Training Loss: {:.4f}, Training Accuracy: {:.4f}\n\n'.format(np.mean(loss_history), (
                    train_correct_total / len(dataset)) * 100))
            ff.close()

        # save the model
        torch.save(model.state_dict(), f'./output_sca_{i}/2_layers/dim_{input_dim}/' + 'model_' + str(epoch + 1) + '.pth')
        print(f"current epoch: {epoch + 1}, time: {time.time() - begin_time}s")

    with open(f'./output_sca_{i}/2_layers/dim_{input_dim}/' + 'train_result.txt', 'a+') as ff:
        ff.write(f'\n\ntotal time: {total_time}')
        ff.close()
    print(f"total time: {time.time() - start_time}s")

def train_3(input_dim, model, i):
    learning_rate = 0.0001
    batch_size = 64
    epochs = 10
    total_time = 0

    # model = CodeCloneDetector(input_dim)
    model = CodeCloneDetector_3(input_dim).to(device)  # GPU

    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    w1 = ProNE.load(f'./metrics_keyword/dim_{input_dim}/embedding.zip')
    dataset = ClonePairDataset('../data/split_data/all.csv', w1, input_dim)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print('Total of dataset: ', len(dataset))
    print('Total of train_loader: ', len(train_loader))
    total_step = len(train_loader)

    model.train()
    size = len(dataset)
    start_time = time.time()
    for epoch in range(epochs):
        print(f"Start {epoch + 1} time is:", datetime.now().strftime("%H:%M"))
        begin_time = time.time()
        train_correct_total = 0
        loss_history = []
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()

            pred = model(X)
            # print('---')
            # print(batch)
            # print(pred)
            # print(y)

            loss = loss_fn(pred, y)


            predicted_labels = torch.argmax(pred, dim=1)

            correct = (predicted_labels == y).sum().item()
            print(f'correct: {correct}')
            train_correct_total += correct
            total = y.size(0)

            batch_accuracy = (predicted_labels == y).sum().item() / y.size(0)
            print(f'Batch {batch}, Accuracy: {batch_accuracy:.4f}')


            loss.backward()

            optimizer.step()


            if (batch + 1) % 100 == 0:
                loss_history.append(loss.item())
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch + 1, epochs, batch + 1,
                                                                                           total_step,
                                                                                           np.mean(loss_history),
                                                                                           (correct / total) * 100))
        print(f"Total correct predictions: {train_correct_total}")

        print('Training Loss: {:.4f}, Training Accuracy: {:.4f}'.format(np.mean(loss_history), (
                train_correct_total / len(dataset)) * 100))
        print(f'Training Time: {time.time() - begin_time}s')
        total_time += time.time() - begin_time
        with open(f'./output_sca_{i}/3_layers/dim_{input_dim}/' + 'train_result.txt', 'a+') as ff:
            ff.write(f'\nModel structure: {model}\n')
            ff.write(f'Dim: {input_dim}\n')
            ff.write(f'Training Time: {time.time() - begin_time}s\n')
            ff.write('Training Loss: {:.4f}, Training Accuracy: {:.4f}\n\n'.format(np.mean(loss_history), (
                    train_correct_total / len(dataset)) * 100))
            ff.close()

        # save the model
        torch.save(model.state_dict(), f'./output_sca_{i}/3_layers/dim_{input_dim}/' + 'model_' + str(epoch + 1) + '.pth')
        print(f"current epoch: {epoch + 1}, time: {time.time() - begin_time}s")

    with open(f'./output_sca_{i}/3_layers/dim_{input_dim}/' + 'train_result.txt', 'a+') as ff:
        ff.write(f'\n\ntotal time: {total_time}')
        ff.close()
    print(f"total time: {time.time() - start_time}s")


def test_1(input_dim, model_load, i):
    batch_size = 64

    # Load the test dataset
    test_vectors_dir = f'../data/split_data/all.csv'
    w1 = ProNE.load(f'./metrics_keyword/dim_{input_dim}/embedding.zip')
    dataset = ClonePairDataset(test_vectors_dir, w1, input_dim)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialization model
    # model_load = CodeCloneDetector(input_dim)
    # model_load = CodeCloneDetector(input_dim).to(device)  # GPU
    model_load.load_state_dict(torch.load(f'./output_sca_{i}/1_layers/dim_{input_dim}/model_10.pth'))
    model_load.eval()

    correct = 0
    total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():  # Gradients are not calculated in the evaluation
        start_time = time.time()
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model_load(X)  # Forward propagation
            predicted_labels = torch.argmax(pred, dim=1)  # Gets the category with the greatest probability

            # Update total
            total += y.size(0)

            # Statistically correct prediction
            correct += (predicted_labels == y).sum().item()

            # Count true positives, false positives and false negatives
            for true, pred in zip(y.cpu().numpy(), predicted_labels.cpu().numpy()):
                if true == 1 and pred == 1:
                    true_positive += 1
                elif true == 0 and pred == 1:
                    false_positive += 1
                elif true == 1 and pred == 0:
                    false_negative += 1
    end_time = time.time()
        # Calculate accuracy, recall and F1 scores
    accuracy = correct / total
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    with open(f'./output_sca_{i}/1_layers/dim_{input_dim}/' + 'test_result.txt', 'a+') as ff:
        ff.write(f'\nDim: {input_dim}\n')
        ff.write(f'test_time: {end_time - start_time}s\n')
        ff.write(f'Test Accuracy: {accuracy:.2f}\n')
        ff.write(f'Test Precision: {precision:.2f}\n')
        ff.write(f'Test Recall: {recall:.2f}\n')
        ff.write(f'Test F1 Score: {f1:.2f}\n\n')
        ff.close()


    print(f'Test Accuracy: {accuracy:.2f}')
    print(f'Test Precision: {precision:.2f}')
    print(f'Test Recall: {recall:.2f}')
    print(f'Test F1 Score: {f1:.2f}')


def test_2(input_dim, model_load, i):
    batch_size = 64

    test_vectors_dir = f'../data/split_data/all.csv'
    w1 = ProNE.load(f'./metrics_keyword/dim_{input_dim}/embedding.zip')
    dataset = ClonePairDataset(test_vectors_dir, w1, input_dim)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # model_load = CodeCloneDetector(input_dim)
    # model_load = CodeCloneDetector(input_dim).to(device)  # GPU
    model_load.load_state_dict(torch.load(f'./output_sca_{i}/2_layers/dim_{input_dim}/model_10.pth'))
    model_load.eval()

    correct = 0
    total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():
        start_time = time.time()
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model_load(X)
            predicted_labels = torch.argmax(pred, dim=1)

            total += y.size(0)


            correct += (predicted_labels == y).sum().item()


            for true, pred in zip(y.cpu().numpy(), predicted_labels.cpu().numpy()):
                if true == 1 and pred == 1:
                    true_positive += 1
                elif true == 0 and pred == 1:
                    false_positive += 1
                elif true == 1 and pred == 0:
                    false_negative += 1
    end_time = time.time()

    accuracy = correct / total
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    with open(f'./output_sca_{i}/2_layers/dim_{input_dim}/' + 'test_result.txt', 'a+') as ff:
        ff.write(f'\nDim: {input_dim}\n')
        ff.write(f'test_time: {end_time - start_time}s\n')
        ff.write(f'Test Accuracy: {accuracy:.2f}\n')
        ff.write(f'Test Precision: {precision:.2f}\n')
        ff.write(f'Test Recall: {recall:.2f}\n')
        ff.write(f'Test F1 Score: {f1:.2f}\n\n')
        ff.close()


    print(f'Test Accuracy: {accuracy:.2f}')
    print(f'Test Precision: {precision:.2f}')
    print(f'Test Recall: {recall:.2f}')
    print(f'Test F1 Score: {f1:.2f}')

def test_3(input_dim, model_load, i):
    batch_size = 64


    test_vectors_dir = f'../data/split_data/all.csv'
    w1 = ProNE.load(f'./metrics_keyword/dim_{input_dim}/embedding.zip')
    dataset = ClonePairDataset(test_vectors_dir, w1, input_dim)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


    # model_load = CodeCloneDetector(input_dim)
    # model_load = CodeCloneDetector(input_dim).to(device)  # GPU
    model_load.load_state_dict(torch.load(f'./output_sca_{i}/3_layers/dim_{input_dim}/model_10.pth'))
    model_load.eval()

    correct = 0
    total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():
        start_time = time.time()
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model_load(X)
            predicted_labels = torch.argmax(pred, dim=1)


            total += y.size(0)


            correct += (predicted_labels == y).sum().item()


            for true, pred in zip(y.cpu().numpy(), predicted_labels.cpu().numpy()):
                if true == 1 and pred == 1:
                    true_positive += 1
                elif true == 0 and pred == 1:
                    false_positive += 1
                elif true == 1 and pred == 0:
                    false_negative += 1
    end_time = time.time()

    accuracy = correct / total
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    with open(f'./output_sca_{i}/3_layers/dim_{input_dim}/' + 'test_result.txt', 'a+') as ff:
        ff.write(f'\nDim: {input_dim}\n')
        ff.write(f'test_time: {end_time - start_time}s\n')
        ff.write(f'Test Accuracy: {accuracy:.2f}\n')
        ff.write(f'Test Precision: {precision:.2f}\n')
        ff.write(f'Test Recall: {recall:.2f}\n')
        ff.write(f'Test F1 Score: {f1:.2f}\n\n')
        ff.close()


    print(f'Test Accuracy: {accuracy:.2f}')
    print(f'Test Precision: {precision:.2f}')
    print(f'Test Recall: {recall:.2f}')
    print(f'Test F1 Score: {f1:.2f}')


if __name__ == '__main__':
    # w1 = ProNE.load('./metrics_keyword/dim_16/embedding.zip')
    # dataset = ClonePairDataset('../data/split_data/train.csv', w1)
    # print(len(dataset))
    # print(dataset[0])
    # print(dataset[1])
    for i in [1, 2, 3]:
        m1 = CodeCloneDetector_1(16).to(device)  # GPU
        m2 = CodeCloneDetector_2(16).to(device)  # GPU
        m3 = CodeCloneDetector_3(16).to(device)  # GPU
        train_1(16, m1, i)
        train_2(16, m2, i)
        train_3(16, m3, i)
        print('-' * 15, 'test:', f'dim_{16}', '-' * 15)
        test_1(16, m1, i)
        test_2(16, m2, i)
        test_3(16, m3, i)

    # m1 = CodeCloneDetector_1(16).to(device)  # GPU
    # m2 = CodeCloneDetector_2(16).to(device)  # GPU
    # m3 = CodeCloneDetector_3(16).to(device)  # GPU
    # test_1(16, m1)
    # test_2(16, m2)
    # test_3(16, m3)
