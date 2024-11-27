import torch
import torch.optim as optim
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix

def train_and_evaluate(
    gcn_1, gcn_2, gcn_3, classifier, features_1, features_2, features_3, labels, runs=10, epochs=100, test_size=0.2, threshold=0.85
):
    accuracies = []
    weighted_f1s = []
    macro_f1s = []
    mccs = []
    cm_sum = None

    for run in range(runs):
        # Split data
        from sklearn.model_selection import train_test_split
        X_train_1, X_test_1, y_train, y_test = train_test_split(features_1.numpy(), labels.numpy(), test_size=test_size, random_state=None)
        X_train_2, X_test_2 = train_test_split(features_2.numpy(), test_size=test_size, random_state=None)
        X_train_3, X_test_3 = train_test_split(features_3.numpy(), test_size=test_size, random_state=None)

        X_train_1, X_test_1 = torch.tensor(X_train_1, dtype=torch.float), torch.tensor(X_test_1, dtype=torch.float)
        X_train_2, X_test_2 = torch.tensor(X_train_2, dtype=torch.float), torch.tensor(X_test_2, dtype=torch.float)
        X_train_3, X_test_3 = torch.tensor(X_train_3, dtype=torch.float), torch.tensor(X_test_3, dtype=torch.float)
        y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

        edge_index_train_1 = create_edge_index_from_features(X_train_1, threshold)
        edge_index_train_2 = create_edge_index_from_features(X_train_2, threshold)
        edge_index_train_3 = create_edge_index_from_features(X_train_3, threshold)

        optimizer = optim.Adam(
            list(gcn_1.parameters())
            + list(gcn_2.parameters())
            + list(gcn_3.parameters())
            + list(classifier.parameters()),
            lr=0.0001,
        )
        criterion = nn.CrossEntropyLoss()

        # Training
        for epoch in range(epochs):
            gcn_1.train()
            gcn_2.train()
            gcn_3.train()
            classifier.train()

            optimizer.zero_grad()

            # Get embeddings
            embedding_1 = gcn_1(X_train_1, edge_index_train_1)
            embedding_2 = gcn_2(X_train_2, edge_index_train_2)
            embedding_3 = gcn_3(X_train_3, edge_index_train_3)

            combined_embeddings = torch.cat([embedding_1, embedding_2, embedding_3], dim=1)

            # Classify
            outputs = classifier(combined_embeddings)
            loss = criterion(outputs, y_train)

            loss.backward()
            optimizer.step()

        # Testing
        gcn_1.eval()
        gcn_2.eval()
        gcn_3.eval()
        classifier.eval()
        with torch.no_grad():
            edge_index_test_1 = create_edge_index_from_features(X_test_1, threshold)
            edge_index_test_2 = create_edge_index_from_features(X_test_2, threshold)
            edge_index_test_3 = create_edge_index_from_features(X_test_3, threshold)

            embedding_1 = gcn_1(X_test_1, edge_index_test_1)
            embedding_2 = gcn_2(X_test_2, edge_index_test_2)
            embedding_3 = gcn_3(X_test_3, edge_index_test_3)

            combined_embeddings = torch.cat([embedding_1, embedding_2, embedding_3], dim=1)
            outputs = classifier(combined_embeddings)
            _, predicted = torch.max(outputs, 1)

        # Metrics
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        accuracies.append(accuracy)
        weighted_f1s.append(f1_score(y_test.numpy(), predicted.numpy(), average="weighted"))
        macro_f1s.append(f1_score(y_test.numpy(), predicted.numpy(), average="macro"))
        mccs.append(matthews_corrcoef(y_test.numpy(), predicted.numpy()))

        # Confusion Matrix
        cm = confusion_matrix(y_test.numpy(), predicted.numpy())
        cm_sum = cm if cm_sum is None else cm_sum + cm

    # Average Metrics
    metrics = {
        "Accuracy": f"{np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}",
        "Weighted F1": f"{np.mean(weighted_f1s):.3f} ± {np.std(weighted_f1s):.3f}",
        "Macro F1": f"{np.mean(macro_f1s):.3f} ± {np.std(macro_f1s):.3f}",
        "MCC": f"{np.mean(mccs):.3f} ± {np.std(mccs):.3f}",
    }
    return metrics
