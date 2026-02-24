import torch
import torch.nn as nn
import torch.optim as optim
import time
from evaluate import evaluate_patch_classifier
from config import NUM_EPOCHS, LEARNING_RATE, CLASS_NAME_MAP


def train(model,
          patch_train_loader,
          patch_val_loader,
          device,
          num_epochs=NUM_EPOCHS,
          save_path=None):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total = 0

        for batch_idx, (imgs, labels) in enumerate(patch_train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            progress = 100 * (batch_idx + 1) / len(patch_train_loader)
            elapsed = time.time() - start_time
            print(
                f"\rEpoch {epoch+1}/{num_epochs} | "
                f"{progress:6.2f}% | "
                f"Batch {batch_idx+1}/{len(patch_train_loader)} | "
                f"Elapsed: {elapsed:.2f} s |"
                f"Loss: {loss.item():.4f}",
                end=""
            )
        train_loss = running_loss / max(total, 1)

        val_metrics = evaluate_patch_classifier(
            model, patch_val_loader, device, criterion
        )
        print()  # Newline after progress bar
        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val P(cones): {val_metrics['cone_macro_precision']:.4f} | "
            f"Val R(cones): {val_metrics['cone_macro_recall']:.4f} | "
            f"Val F1(cones): {val_metrics['cone_macro_f1']:.4f}"

        )

        for c in CLASS_NAME_MAP:
            m = val_metrics["per_class"][c]
            print(
                f"  {CLASS_NAME_MAP[c]:10s} | "
                f"P={m['precision']:.4f} "
                f"R={m['recall']:.4f} "
                f"F1={m['f1']:.4f}"
            )

    # Save only if path provided
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f"Model weights saved to {save_path}")